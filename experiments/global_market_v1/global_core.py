"""Pooled log-XGBoost with frozen Country categories and per-country recursion."""
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from dependencies import SERIES_KEY, core


FEATURE_COLUMNS = core.FEATURE_COLUMNS + ['Country']
NUMERIC_FEATURES = core.NUMERIC_FEATURES.copy()
CATEGORICAL_FEATURES = ['Brand', 'Model', 'Country']
PREDICTION_COLUMNS = core.PREDICTION_COLUMNS.copy()
STATUS_COLUMNS = core.STATUS_COLUMNS.copy()
FEATURE_DTYPES = {name: 'category' if name in CATEGORICAL_FEATURES else
                  'int32' if name == 'month' else 'float64' for name in FEATURE_COLUMNS}


def _validate_schema(schema):
    if (schema.get('scope') != 'global' or schema.get('feature_columns') != FEATURE_COLUMNS
            or schema.get('feature_dtypes') != FEATURE_DTYPES):
        raise ValueError('Global models require the original 11 features followed by categorical Country')
    for name, expected in (('target_transform', 'log1p'), ('prediction_inverse', 'expm1'),
                           ('prediction_unit', 'sales')):
        if schema.get(name) != expected:
            raise ValueError(f'Unexpected global transform or unit: {name}')
    countries = schema.get('countries')
    if (not isinstance(countries, list) or not countries
            or not all(isinstance(value, str) and value.strip() for value in countries)
            or countries != sorted(set(countries))):
        raise ValueError('Global countries must be sorted unique nonempty labels')
    levels = schema.get('categorical_levels', {})
    for name in CATEGORICAL_FEATURES:
        values = levels.get(name)
        if (not isinstance(values, list) or not values
                or not all(isinstance(value, str) and value.strip() for value in values)
                or values != sorted(set(values))):
            raise ValueError(f'Invalid frozen global categories: {name}')
    if levels['Country'] != countries:
        raise ValueError('Country categories disagree with the global country list')
    seen = schema.get('seen_series')
    if (not isinstance(seen, list) or not seen
            or any(not isinstance(key, list) or len(key) != 3 for key in seen)):
        raise ValueError('Global schema requires exact learned Country/Brand/Model keys')
    for key in seen:
        if any(not isinstance(value, str) or value not in levels[name]
               for name, value in zip(SERIES_KEY, key)):
            raise ValueError('Learned global series disagree with frozen categories')
    if seen != [list(key) for key in sorted({tuple(key) for key in seen})]:
        raise ValueError('Learned global series must be sorted and unique')
    for index, name in enumerate(SERIES_KEY):
        if levels[name] != sorted({key[index] for key in seen}):
            raise ValueError('Frozen categories must come only from learned series')
    train_end = core._month(schema['train_end'])
    start, end = (core._month(schema[name]) for name in ('training_target_start', 'training_target_end'))
    count = schema.get('training_rows')
    if (not isinstance(count, int) or isinstance(count, bool) or count < 1
            or not start <= end <= train_end):
        raise ValueError('Invalid global training row count or target range')


def _feature_frame(frame, schema):
    features = frame[FEATURE_COLUMNS].copy()
    for name in NUMERIC_FEATURES:
        features[name] = features[name].astype(FEATURE_DTYPES[name])
    for name in CATEGORICAL_FEATURES:
        features[name] = pd.Categorical(features[name], categories=schema['categorical_levels'][name])
        if features[name].isna().any():
            raise ValueError(f'Unknown global category: {name}')
    return features


def fit_global(training_frame, parameters, train_end, n_jobs=4):
    """Fit one pooled model to eligible targets at or before the supplied cutoff."""
    train_end = core._month(train_end)
    rows = training_frame.loc[training_frame.Month.le(train_end)].copy()
    if rows.empty:
        raise ValueError('NO_TRAINING_ROWS')
    if rows.duplicated(SERIES_KEY + ['Month']).any():
        raise ValueError('Duplicate learned series/month in pooled training rows')
    if rows[SERIES_KEY].isna().any().any():
        raise ValueError('Missing pooled training series identifier')
    rows = rows.sort_values(SERIES_KEY + ['Month']).reset_index(drop=True)
    fixed = {'objective': 'reg:squarederror', 'tree_method': 'hist', 'enable_categorical': True,
             'device': 'cpu', 'random_state': 42, 'subsample': 1.0, 'colsample_bytree': 1.0}
    for name, value in fixed.items():
        if name in parameters and parameters[name] != value:
            raise ValueError(f'Cannot change fixed global parameter: {name}')
    if parameters.get('early_stopping_rounds') is not None:
        raise ValueError('Global candidates use fixed tree counts without early stopping')
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1:
        raise ValueError('CPU n_jobs must be a positive integer')
    settings = {'max_depth': 3, 'learning_rate': .03, 'n_estimators': 200,
                **parameters, **fixed, 'n_jobs': n_jobs}
    schema = {'scope': 'global', 'countries': sorted(rows.Country.unique().tolist()),
              'feature_columns': FEATURE_COLUMNS.copy(), 'feature_dtypes': FEATURE_DTYPES.copy(),
              'target_transform': 'log1p', 'prediction_inverse': 'expm1', 'prediction_unit': 'sales',
              'model_output_unit': 'log1p_sales', 'input_sales_unit': 'sales',
              'categorical_levels': {name: sorted(rows[name].unique().tolist()) for name in CATEGORICAL_FEATURES},
              'seen_series': rows[SERIES_KEY].drop_duplicates().values.tolist(), 'train_end': str(train_end),
              'training_rows': len(rows), 'training_target_start': str(rows.Month.min()),
              'training_target_end': str(rows.Month.max())}
    _validate_schema(schema)
    features = _feature_frame(rows, schema)
    target = rows.Sales.to_numpy(dtype=float)
    if (not np.isfinite(features[NUMERIC_FEATURES].to_numpy(dtype=float)).all()
            or not np.isfinite(target).all() or (target < 0).any()):
        raise ValueError('Nonfinite pooled features or invalid training sales')
    model = XGBRegressor(**settings)
    model.fit(features, np.log1p(target))
    schema['parameters'] = {name: 'NaN' if isinstance(value, float) and np.isnan(value) else value
                            for name, value in model.get_params().items()}
    schema['booster_config'] = json.loads(model.get_booster().save_config())
    model.get_booster().set_attr(daizen_feature_schema_sha256=core._schema_digest(schema))
    return model, schema


class _CountryView:
    """Return raw global log predictions to the preserved recursive implementation."""
    def __init__(self, model, schema, country):
        self.model, self.schema, self.country = model, schema, country

    def predict(self, frame):
        pooled = frame.copy()
        pooled['Country'] = self.country
        return self.model.predict(_feature_frame(pooled, self.schema))


def _country_schema(schema, country):
    # This ephemeral view is never saved and never changes the original core/schema.
    return {**schema, 'country': country, 'feature_columns': core.FEATURE_COLUMNS.copy(),
            'feature_dtypes': {name: FEATURE_DTYPES[name] for name in core.FEATURE_COLUMNS},
            'categorical_levels': {name: schema['categorical_levels'][name].copy() for name in ('Brand', 'Model')},
            'seen_series': [key.copy() for key in schema['seen_series'] if key[0] == country]}


def forecast_global(model, history, origin, horizon, schema):
    """Keep each country/series history separate while sharing one fitted model."""
    _validate_schema(schema)
    origin = core._month(origin)
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError('Forecast horizon must be a positive integer')
    if core._month(schema['train_end']) > origin:
        raise ValueError('Global training cutoff is later than the forecast origin')
    if history[SERIES_KEY + ['Month']].isna().any().any():
        raise ValueError('Missing forecast series identifier or month')
    if history.Month.gt(origin).any():
        raise ValueError('Future actual observations cannot enter global forecast history')
    if history.duplicated(SERIES_KEY + ['Month']).any():
        raise ValueError('Duplicate global forecast series/month')
    predictions, statuses = [], []
    for country, country_history in history.groupby('Country', sort=True):
        view = _CountryView(model, schema, country)
        predicted, state = core.forecast_recursive(view, country_history, origin, horizon,
                                                   _country_schema(schema, country))
        if not predicted.empty:
            predictions.append(predicted)
        statuses.append(state)
    return (pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=PREDICTION_COLUMNS),
            pd.concat(statuses, ignore_index=True) if statuses else pd.DataFrame(columns=STATUS_COLUMNS))


def _validate_model_schema(model, schema):
    booster = model.get_booster()
    if (booster.feature_names != FEATURE_COLUMNS
            or booster.attr('daizen_feature_schema_sha256') != core._schema_digest(schema)
            or booster.feature_types[-3:] != ['c', 'c', 'c']):
        raise ValueError('Global model and schema/category hash do not match')


def save_global_model(model, schema, path):
    """Create a UBJSON model and bound global schema without overwriting artifacts."""
    _validate_schema(schema)
    path = Path(path)
    if path.suffix.lower() != '.ubj':
        raise ValueError('Global categorical models must be saved as .ubj')
    schema_path = path.with_name('feature_schema.json')
    if path.exists() or schema_path.exists():
        raise FileExistsError('Existing global model or feature schema cannot be overwritten')
    _validate_model_schema(model, schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.model_staging_', dir=path.parent) as temporary:
        staged = Path(temporary) / 'model.ubj'
        model.save_model(staged)
        schema_text = json.dumps(schema, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
        created = []
        try:
            with path.open('xb') as destination:
                created.append(path)
                destination.write(staged.read_bytes())
            with schema_path.open('x', encoding='utf-8') as destination:
                created.append(schema_path)
                destination.write(schema_text)
        except BaseException:
            for artifact in created:
                artifact.unlink(missing_ok=True)
            raise


def load_global_model(path):
    """Load a bound pooled model and restore only its recorded CPU runtime settings."""
    path = Path(path)
    schema = json.loads(path.with_name('feature_schema.json').read_text(encoding='utf-8'))
    _validate_schema(schema)
    model = XGBRegressor()
    model.load_model(path)
    _validate_model_schema(model, schema)
    runtime = schema.get('parameters', {})
    n_jobs = runtime.get('n_jobs')
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1 or runtime.get('device') != 'cpu':
        raise ValueError('Invalid saved global CPU runtime settings')
    model.set_params(n_jobs=n_jobs, device='cpu')
    return model, schema
