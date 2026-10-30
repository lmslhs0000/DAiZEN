import hashlib
import json
import sys
import shutil
import calendar_features as cf
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/global_market_v1'))
import global_core as gc
from dependencies import core, helpers, SERIES_KEY, evaluate_forecast
sys.path.insert(0, str(ROOT / 'experiments/market_context_v1'))
from context_features import _validated_copy
# Restore the local module directory after loading preserved dependencies.
sys.path.insert(0, str(Path(__file__).resolve().parent))

KEY = SERIES_KEY
CAPACITIES = {'calendar_l2': {'max_depth': 5, 'n_estimators': 200}}
VARIANTS = list(CAPACITIES)
OBJECTIVE = 'reg:squarederror'
EXTRA_LAGS = [4, 5, 7, 8, 9, 10, 11, 13, 18, 24]
EXTRA = ([f'lag_{k}' for k in EXTRA_LAGS] + [f'relative_lag_{k}' for k in range(1, 13)]
         + [f'relative_std_{k}' for k in [3, 6, 12]]
         + ['zero_share_3', 'zero_share_12', 'change_recent3_prior9', 'change_recent3_previous3',
            'change_last_recent3', 'change_year_1', 'change_year_3', 'log_slope_12',
            'history_observed_24', 'series_age_months', 'month_sin', 'month_cos'])
EXTRA.extend(cf.COLUMNS)
FEATURE_COLUMNS = gc.FEATURE_COLUMNS + EXTRA
NUMERIC = gc.NUMERIC_FEATURES + EXTRA
BINDING = 'daizen_calendar_exposure_schema_sha256'


def _source_hashes():
    return {name: helpers.file_sha256(Path(__file__).parent / name)
            for name in ['calendar_core.py', 'calendar_features.py']}


def _verify_sources(model, schema):
    directory = Path(getattr(model, '_daizen_source_code', Path(__file__).parent))
    for name, checksum in schema['bundle_source_hashes'].items():
        if Path(name).name != name or helpers.file_sha256(directory / name) != checksum:
            raise ValueError('Stored source code hash mismatch')


def observations(data, cutoff, reject_future=False):
    frame = _validated_copy(data)
    cutoff = core._month(cutoff)
    if reject_future and frame.Month.gt(cutoff).any():
        raise ValueError('Future observations cannot enter recursion')
    return frame.loc[frame.Month.between(pd.Period('2016-01', 'M'), cutoff)].sort_values(KEY + ['Month']).reset_index(drop=True)


def make_features(windows, keys, target_months, ages):
    windows = np.asarray(windows, dtype=float)
    if windows.shape != (len(keys), 24):
        raise ValueError('Feature history must have 24 calendar positions')
    recent = windows[:, -12:]
    if not np.isfinite(recent).all() or (windows[np.isfinite(windows)] < 0).any() or np.isinf(windows).any():
        raise ValueError('Recent 12 observations must be finite and nonnegative')
    months = pd.PeriodIndex(target_months, freq='M')
    if len(months) != len(keys) or len(ages) != len(keys):
        raise ValueError('Feature key/calendar lengths differ')
    table = pd.DataFrame(keys, columns=KEY)
    table['Month'] = months
    for lag in [1, 2, 3, 6, 12] + EXTRA_LAGS:
        table[f'lag_{lag}'] = windows[:, -lag]
    for length in [3, 6, 12]:
        table[f'rolling_mean_{length}'] = np.sum(recent[:, -length:] / length, axis=1)
        table[f'relative_std_{length}'] = np.std(recent[:, -length:], axis=1) / np.maximum(table[f'rolling_mean_{length}'], 1.)
    level = np.log1p(table.rolling_mean_12.to_numpy())
    for lag in range(1, 13):
        table[f'relative_lag_{lag}'] = np.log1p(recent[:, -lag]) - level
    table['zero_share_3'] = (recent[:, -3:] == 0).mean(axis=1)
    table['zero_share_12'] = (recent == 0).mean(axis=1)
    recent3 = table.rolling_mean_3.to_numpy()
    table['change_recent3_prior9'] = np.log1p(recent3) - np.log1p(np.sum(recent[:, :9] / 9, axis=1))
    table['change_recent3_previous3'] = np.log1p(recent3) - np.log1p(np.sum(recent[:, -6:-3] / 3, axis=1))
    table['change_last_recent3'] = np.log1p(recent[:, -1]) - np.log1p(recent3)
    table['change_year_1'] = np.log1p(windows[:, -1]) - np.log1p(windows[:, -13])
    table['change_year_3'] = np.log1p(recent3) - np.log1p(np.sum(windows[:, -15:-12] / 3, axis=1))
    centered_time = np.arange(12, dtype=float) - 5.5
    table['log_slope_12'] = np.sum(np.log1p(recent) * centered_time, axis=1) / np.sum(centered_time ** 2)
    table['history_observed_24'] = np.isfinite(windows).sum(axis=1).astype('int32')
    table['series_age_months'] = np.asarray(ages, dtype='int32')
    table['month'] = months.month.astype('int32')
    table['month_sin'] = np.sin(2 * np.pi * months.month / 12)
    table['month_cos'] = np.cos(2 * np.pi * months.month / 12)
    calendars = {month: cf.month_features(str(month)) for month in months.unique()}
    for name in cf.COLUMNS:
        table[name] = [calendars[month][name] for month in months]
    return table[KEY + ['Month'] + [c for c in FEATURE_COLUMNS if c not in KEY]]


def build_training_data(data, cutoff):
    observed = observations(data, cutoff)
    base, audit = core.build_training_frame(observed, cutoff)
    tables = []
    eligible = {key: group for key, group in base.groupby(KEY, sort=True)}
    for key, group in observed.groupby(KEY, sort=True):
        if key not in eligible:
            continue
        targets = eligible[key][['Month', 'Sales']]
        first = group.Month.min()
        calendar = pd.period_range(first - 24, core._month(cutoff), freq='M')
        values = group.set_index('Month').Sales.reindex(calendar).to_numpy(float)
        matrix = np.lib.stride_tricks.sliding_window_view(values, 24)
        positions = np.asarray([m.ordinal - calendar[0].ordinal - 24 for m in targets.Month], int)
        windows = matrix[positions]
        ages = [m.ordinal - first.ordinal + 1 for m in targets.Month]
        table = make_features(windows, [key] * len(targets), targets.Month, ages)
        table['Sales'] = targets.Sales.to_numpy(float)
        tables.append(table)
    columns = KEY + ['Month', 'Sales'] + [c for c in FEATURE_COLUMNS if c not in KEY]
    result = pd.concat(tables, ignore_index=True)[columns] if tables else pd.DataFrame(columns=columns)
    result = result.sort_values(KEY + ['Month']).reset_index(drop=True)
    if len(result) != len(base):
        raise ValueError('Extended features cannot remove original training rows')
    pd.testing.assert_frame_equal(result[base.columns], base, check_exact=False, rtol=1e-12, atol=1e-9)
    return result, audit


def transform_target(actual, anchor, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown variant')
    actual, anchor = np.asarray(actual, float), np.asarray(anchor, float)
    if not np.isfinite(actual).all() or (actual < 0).any() or not np.isfinite(anchor).all() or (anchor < 0).any():
        raise ValueError('Invalid actual or seasonal anchor')
    return np.log1p(actual)


def inverse_target(raw, anchor, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown variant')
    transformed = np.asarray(raw, float)
    with np.errstate(over='ignore', invalid='ignore'):
        return np.maximum(0., np.expm1(transformed))


def validate_schema(schema):
    variant = schema.get('variant')
    if (variant not in VARIANTS or schema.get('feature_columns') != FEATURE_COLUMNS
            or schema.get('feature_policy') != 'own_past_24_plus_calendar_v1'
            or schema.get('scope') != 'global_calendar_exposure'):
        raise ValueError('Invalid rich-history model schema')
    if (schema.get('calendar_definition') != cf.DEFINITION
            or schema.get('bundle_source_hashes') != _source_hashes()):
        raise ValueError('Stored calendar definition or source changed')
    expected = 'log1p_sales'
    if (schema.get('capacity') != CAPACITIES[variant]
            or schema.get('objective_id') != OBJECTIVE
            or schema.get('intercept_policy') != 'mean_log_training_labels'
            or any(name in schema for name in ['asymmetry_tau', 'transition_width_log', 'loss_source_sha256'])):
        raise ValueError('Capacity or objective configuration disagrees with variant')
    parameters = schema.get('parameters')
    if parameters is not None:
        fixed = {**CAPACITIES[variant], 'learning_rate': .1, 'n_jobs': 4, 'objective': OBJECTIVE,
                 'tree_method': 'hist', 'enable_categorical': True, 'device': 'cpu',
                 'random_state': 42, 'subsample': 1., 'colsample_bytree': 1.,
                 'disable_default_eval_metric': True, 'base_score': schema['base_score']}
        if any(parameters.get(name) != value for name, value in fixed.items()) or parameters.get('early_stopping_rounds') is not None:
            raise ValueError('Stored parameters disagree with fixed capacity experiment')
    if (not isinstance(schema.get('base_score'), (int, float))
            or isinstance(schema['base_score'], bool) or not np.isfinite(schema['base_score'])
            or schema['base_score'] < 0):
        raise ValueError('Training-only log-sales intercept must be finite and nonnegative')
    if (schema.get('target_transform') != expected or schema.get('prediction_unit') != 'sales'
            or schema.get('prediction_inverse') != 'expm1'
            or schema.get('feature_dtypes') != {c: 'category' if c in KEY else 'int32' if c == 'month' else 'float64' for c in FEATURE_COLUMNS}):
        raise ValueError('Target representation does not match variant')
    gc._validate_schema({**schema, 'scope': 'global', 'feature_columns': gc.FEATURE_COLUMNS,
                         'feature_dtypes': gc.FEATURE_DTYPES, 'target_transform': 'log1p',
                         'prediction_inverse': 'expm1'})


def feature_frame(rows, schema):
    validate_schema(schema)
    frame = rows[FEATURE_COLUMNS].copy()
    for name in NUMERIC:
        frame[name] = frame[name].astype('int32' if name == 'month' else 'float64')
    if not np.isfinite(frame[gc.NUMERIC_FEATURES].to_numpy(float)).all() or np.isinf(frame[EXTRA].to_numpy(float)).any():
        raise ValueError('Invalid numeric model input')
    for name in gc.CATEGORICAL_FEATURES:
        frame[name] = pd.Categorical(frame[name], categories=schema['categorical_levels'][name])
        if frame[name].isna().any():
            raise ValueError('Unlearned category')
    return frame


def fit(rows, variant, cutoff):
    if variant not in VARIANTS:
        raise ValueError('Unknown variant')
    cutoff = core._month(cutoff)
    training = rows.loc[rows.Month.le(cutoff)].sort_values(KEY + ['Month']).reset_index(drop=True)
    if training.empty or training.duplicated(KEY + ['Month']).any():
        raise ValueError('Missing or duplicate training rows')
    calendars = {month: cf.month_features(str(month)) for month in training.Month.unique()}
    for name in cf.COLUMNS:
        expected = np.asarray([calendars[month][name] for month in training.Month], dtype=float)
        if not np.array_equal(training[name].to_numpy(float), expected):
            raise ValueError('Training calendar values disagree with target month')
    settings = {**CAPACITIES[variant], 'learning_rate': .1, 'n_jobs': 4,
                'objective': OBJECTIVE,
                'base_score': float(np.log1p(training.Sales.to_numpy(float)).mean()),
                'disable_default_eval_metric': True,
                'tree_method': 'hist', 'enable_categorical': True, 'device': 'cpu',
                'random_state': 42, 'subsample': 1., 'colsample_bytree': 1.}
    schema = {'scope': 'global_calendar_exposure', 'variant': variant, 'capacity': CAPACITIES[variant].copy(),
              'objective_id': OBJECTIVE,
              'calendar_definition': json.loads(json.dumps(cf.DEFINITION)),
              'bundle_source_hashes': _source_hashes(),
              'intercept_policy': 'mean_log_training_labels', 'base_score': settings['base_score'],
              'feature_columns': FEATURE_COLUMNS.copy(),
              'feature_dtypes': {c: 'category' if c in gc.CATEGORICAL_FEATURES else 'int32' if c == 'month' else 'float64' for c in FEATURE_COLUMNS},
              'feature_policy': 'own_past_24_plus_calendar_v1',
              'target_transform': 'log1p_sales',
              'prediction_inverse': 'expm1',
              'prediction_unit': 'sales', 'countries': sorted(training.Country.unique().tolist()),
              'categorical_levels': {c: sorted(training[c].unique().tolist()) for c in gc.CATEGORICAL_FEATURES},
              'seen_series': training[KEY].drop_duplicates().values.tolist(), 'train_end': str(cutoff),
              'training_rows': len(training), 'training_target_start': str(training.Month.min()),
              'training_target_end': str(training.Month.max())}
    inputs = feature_frame(training, schema)
    target = transform_target(training.Sales, training.lag_12, variant)
    model = XGBRegressor(**settings)
    model.fit(inputs, target)
    schema['parameters'] = {k: 'NaN' if isinstance(v, float) and np.isnan(v) else v
                            for k, v in model.get_params().items()}
    schema['code_source_sha256'] = helpers.file_sha256(Path(__file__))
    schema['fit_booster_config'] = json.loads(model.get_booster().save_config())
    schema['booster_objective_for_inference'] = 'reg:squarederror'
    schema['target_range'] = [float(target.min()), float(target.max())]
    canonical = training[KEY + ['Month', 'Sales'] + [c for c in FEATURE_COLUMNS if c not in KEY]].copy()
    canonical['Month'] = canonical.Month.astype(str)
    for name in ['Sales'] + [c for c in FEATURE_COLUMNS if c not in KEY]:
        canonical[name] = canonical[name].astype('int32' if name == 'month' else 'float64')
    schema['training_input_sha256'] = hashlib.sha256(pd.util.hash_pandas_object(canonical, index=False).to_numpy('uint64').tobytes()).hexdigest()
    schema['input_sales_unit'] = 'sales'
    schema['model_output_unit'] = 'log1p_sales'
    model.get_booster().set_attr(**{BINDING: core._schema_digest(schema)})
    return model, schema


def forecast(model, history, origin, horizon, schema):
    validate_schema(schema)
    origin = core._month(origin)
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1 or core._month(schema['train_end']) > origin:
        raise ValueError('Invalid horizon or training cutoff')
    observed = observations(history, origin, reject_future=True)
    seen = {tuple(key) for key in schema['seen_series']}
    keys, buffers, ages, indices, states = [], [], [], [], []
    required = pd.period_range(origin - 11, origin, freq='M')
    extended = pd.period_range(origin - 23, origin, freq='M')
    for key, group in observed.groupby(KEY, sort=True):
        monthly = group.set_index('Month').Sales
        status = ('INSUFFICIENT_HISTORY' if monthly.index.min() > required[0]
                  else 'INCOMPLETE_HISTORY' if monthly.reindex(required).isna().any() else 'OK')
        if status == 'OK' and key not in seen:
            status = 'UNSEEN_SERIES'
        states.append({**dict(zip(KEY, key)), 'prediction_status': status, 'reason': '' if status == 'OK' else status})
        if status == 'OK':
            keys.append(key)
            buffers.append(monthly.reindex(extended).to_numpy(float))
            ages.append(origin.ordinal - monthly.index.min().ordinal + 2)
            indices.append(len(states) - 1)
    if not keys:
        return pd.DataFrame(columns=core.PREDICTION_COLUMNS), pd.DataFrame(states, columns=core.STATUS_COLUMNS)
    buffers = np.asarray(buffers, float)
    values = np.full((len(keys), horizon), np.nan)
    active = np.arange(len(keys))
    for step in range(horizon):
        if not len(active):
            break
        target = origin + step + 1
        batch = make_features(buffers[active], [keys[i] for i in active], [target] * len(active),
                              [ages[i] + step for i in active])
        try:
            raw = np.asarray(model.predict(feature_frame(batch, schema)), float)
            if raw.shape != (len(active),):
                raise ValueError('Model output shape mismatch')
            prediction = inverse_target(raw, batch.lag_12, schema['variant'])
        except Exception as exc:
            for i in active:
                states[indices[i]].update(prediction_status='PREDICTION_ERROR', reason=f'{target}: {exc}')
            break
        finite = np.isfinite(raw) & np.isfinite(prediction)
        for i in active[~finite]:
            states[indices[i]].update(prediction_status='PREDICTION_ERROR', reason=f'{target}: Nonfinite inverse prediction')
        good = active[finite]
        values[good, step] = prediction[finite]
        buffers[good, :-1] = buffers[good, 1:]
        buffers[good, -1] = prediction[finite]
        active = good
    records = [{**dict(zip(KEY, key)), 'Month': origin + step + 1, 'horizon': step + 1, 'prediction': value}
               for i, key in enumerate(keys) if states[indices[i]]['prediction_status'] == 'OK'
               for step, value in enumerate(values[i])]
    return pd.DataFrame(records, columns=core.PREDICTION_COLUMNS), pd.DataFrame(states, columns=core.STATUS_COLUMNS)


def bound_model(model, schema):
    validate_schema(schema)
    _verify_sources(model, schema)
    booster = model.get_booster()
    if booster.feature_names != FEATURE_COLUMNS or booster.attr(BINDING) != core._schema_digest(schema):
        raise ValueError('Model/schema hash binding mismatch')
    if (schema.get('booster_objective_for_inference') != OBJECTIVE
            or schema.get('parameters', {}).get('objective') != OBJECTIVE
            or schema.get('code_source_sha256') != helpers.file_sha256(Path(__file__))):
        raise ValueError('Stored code or inference contract mismatch')
    if (schema.get('input_sales_unit') != 'sales' or schema.get('model_output_unit') != 'log1p_sales'
            or len(schema.get('training_input_sha256', '')) != 64):
        raise ValueError('Stored training inputs or units are invalid')
    config = json.loads(booster.save_config())['learner']
    fitted = schema['fit_booster_config']['learner']
    train_parameters = fitted['gradient_booster']['tree_train_param']
    if (config['objective']['name'] != OBJECTIVE or fitted['objective']['name'] != OBJECTIVE
            or int(train_parameters['max_depth']) != schema['capacity']['max_depth']
            or int(fitted['gradient_booster']['gbtree_model_param']['num_trees']) != schema['capacity']['n_estimators']
            or booster.num_boosted_rounds() != schema['capacity']['n_estimators']):
        raise ValueError('Actual objective or trained capacity disagrees with schema')


def save(model, schema, path):
    path = Path(path)
    schema_path = path.with_name('feature_schema.json')
    if path.exists() or schema_path.exists():
        raise FileExistsError('Existing model artifacts cannot be overwritten')
    if path.suffix != '.ubj':
        raise ValueError('Model must use UBJSON')
    bound_model(model, schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)
    directory = path.parent / 'source_code'
    directory.mkdir(exist_ok=False)
    for name in schema['bundle_source_hashes']:
        shutil.copy2(Path(__file__).parent / name, directory / name)
    helpers.write_json(schema_path, schema)


def load(path):
    path = Path(path)
    schema = json.loads(path.with_name('feature_schema.json').read_text(encoding='utf-8'))
    model = XGBRegressor()
    model.load_model(path)
    model._daizen_source_code = str(path.parent / 'source_code')
    bound_model(model, schema)
    model.set_params(n_jobs=4, device='cpu')
    return model, schema
