"""rich_cb_yoy_lcg seed 44 코어(2026-10-05 사전 고정 (dq)): cblcg_core.py 사본에서 CatBoost random_seed만 44로 바꿨다. 아래는 원래 설명 — rich_cb_yoy_lcg 코어(2026-10-04 사전 고정 (dg)): cbyoyall_core.py 사본에서 수명 특징만 전 시장판(lifecycle_features_g: 시장별 조사 사건, 곧 나올 신세대 포함 4개)으로 바꿨다. 아래는 원래 설명 — rich_cb_yoy 코어(2026-10-04 사전 고정 (dc)): cball_core.py 사본에서 정답만 작년 대비 변화로 바꿨다 — 정답 = log1p(판매) − log1p(12개월 전 판매), 예측 = expm1(출력 + log1p(12개월 전 값)), 0 하한. 아래는 원래 설명 — rich_cb_all 코어(2026-10-03 사전 고정 (cl)): rich_all_v1/calall_core.py 사본에서 학습기만 CatBoost로 바꿨다(같은 학습 행·입력·로그 정답·재귀 규칙; 깊이 5·나무 200·학습률 0.1·RMSE·seed 42, 범주 입력은 CatBoost 범주형, 저장은 .cbm). 아래는 원래 설명 — rich_cal_all 코어(2026-10-03 사전 고정 (ci)): cvwall_core.py 사본에서 학습 표본 가중치만 뺐다(rich_cal과 같은 비가중 학습 + 신호 12개). 아래는 원래 설명 — rich calendar_vw_all 코어(2026-10-03 사전 고정 (cc)·(cd)): ebl_core.py 사본에 남은 중국·외부 신호를 더해 신차 비중·브랜드 흐름·수명 사건·브랜드 국적 점유율·공식 시장 판매 12개를 모두 쓴다. 아래는 원래 설명 — rich_cal_vw_ebl 코어(2026-10-03 사전 고정 (bi)): rich_eb_v1/eb_core.py 사본에 (l2) 중국 수명 특징 3개
(lifecycle_features, rich_lcvw_v1과 같은 사건·대상 표 복사본)를 더했다. 아래는 eb_core 설명 — entrant_v1/ent_core.py 사본에 브랜드 흐름 2개(brand_momentum_v1의
brand_features.COLUMNS)를 함께 더했다(stats = feature_stats()의 사전). 아래는 ent_core 설명 — rich_vw_v1/richvw_core.py(rich_cal_vw) 사본에 시장 신차 비중 2개
(entrant_features.COLUMNS)를 더했다. 기준 시점은 학습 행 = 대상월 직전 달, 재귀 예측 = 기준월 고정(market_momentum_v1와 같은 방식).
바꾼 곳은 특징·변형·스키마 이름·소스 해시뿐이다. 원래 설명 — rich_cal_vw: calendar_core 사본에 학습 표본 가중치
(차종의 학습 행 평균 판매량 ÷ 차종 평균, 평균 1). 정답은 로그 판매량 그대로.
"""

import hashlib
import json
import sys
import shutil
import calendar_features as cf
import entrant_features as efe
import brand_features as bfe
import origin_features as ofe
import official_features as offe
import lifecycle_features_g as lfe
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/global_market_v1'))
import global_core as gc
from dependencies import core, helpers, SERIES_KEY, evaluate_forecast
sys.path.insert(0, str(ROOT / 'experiments/market_context_v1'))
from context_features import _validated_copy
# Restore the local module directory after loading preserved dependencies.
sys.path.insert(0, str(Path(__file__).resolve().parent))

KEY = SERIES_KEY
CAPACITIES = {'catboost_yoy_lcg_s44': {'max_depth': 5, 'n_estimators': 200}}
VARIANTS = list(CAPACITIES)
OBJECTIVE = 'RMSE'
MODEL_FILE = 'model.cbm'  # forecast_rich49가 저장 파일 이름으로 씀
EXTRA_LAGS = [4, 5, 7, 8, 9, 10, 11, 13, 18, 24]
EXTRA = ([f'lag_{k}' for k in EXTRA_LAGS] + [f'relative_lag_{k}' for k in range(1, 13)]
         + [f'relative_std_{k}' for k in [3, 6, 12]]
         + ['zero_share_3', 'zero_share_12', 'change_recent3_prior9', 'change_recent3_previous3',
            'change_last_recent3', 'change_year_1', 'change_year_3', 'log_slope_12',
            'history_observed_24', 'series_age_months', 'month_sin', 'month_cos'])
EXTRA.extend(cf.COLUMNS)
EXTRA.extend(efe.COLUMNS)
EXTRA.extend(bfe.COLUMNS)
EXTRA.extend(lfe.COLUMNS)
EXTRA.extend(ofe.COLUMNS)
EXTRA.extend(offe.COLUMNS)
FEATURE_COLUMNS = gc.FEATURE_COLUMNS + EXTRA
NUMERIC = gc.NUMERIC_FEATURES + EXTRA
BINDING = 'daizen_catboost_yoy_lcg_s44_schema_sha256'


def _source_hashes():
    return {name: helpers.file_sha256(Path(__file__).parent / name)
            for name in ['cblcg_s44_core.py', 'calendar_features.py', 'entrant_features.py', 'brand_features.py', 'lifecycle_features.py',
                         'lifecycle_events_l2.csv', 'lifecycle_features_g.py', 'lifecycle_events_global.csv',
                         'lifecycle_targets_global.csv', 'origin_features.py', 'brand_origin.csv', 'official_features.py']}


def _verify_sources(model, schema):
    directory = Path(getattr(model, '_daizen_source_code', Path(__file__).parent))
    for name, checksum in schema['bundle_source_hashes'].items():
        if Path(name).name != name or helpers.file_sha256(directory / name) != checksum:
            raise ValueError('Stored source code hash mismatch')


def cb_settings(variant):
    """(cl) CatBoost 고정 설정 — XGBoost rich와 같은 용량(깊이 5, 나무 200, 학습률 0.1)."""
    return {'depth': CAPACITIES[variant]['max_depth'], 'iterations': CAPACITIES[variant]['n_estimators'],
            'learning_rate': 0.1, 'loss_function': OBJECTIVE, 'random_seed': 44, 'thread_count': 4,
            'boost_from_average': True, 'allow_writing_files': False, 'verbose': False,
            'cat_features': list(gc.CATEGORICAL_FEATURES)}


def cb_frame(frame):
    """CatBoost 범주형은 문자열로 넘긴다(검사가 끝난 feature_frame 결과를 받음)."""
    out = frame.copy()
    for name in gc.CATEGORICAL_FEATURES:
        out[name] = out[name].astype(str)
    return out


def feature_stats(observed):
    """기준 시점 특징에 쓰는 통계: 신차 비중용 시장 표와 브랜드 흐름용 (시장, 브랜드) 합계."""
    return {'entrant': efe.entrant_stats(observed), 'brand': bfe.brand_totals(observed),
            'origin': ofe.origin_stats(observed), 'official': offe.official_stats(observed)}


def observations(data, cutoff, reject_future=False):
    frame = _validated_copy(data)
    cutoff = core._month(cutoff)
    if reject_future and frame.Month.gt(cutoff).any():
        raise ValueError('Future observations cannot enter recursion')
    return frame.loc[frame.Month.between(pd.Period('2016-01', 'M'), cutoff)].sort_values(KEY + ['Month']).reset_index(drop=True)


def make_features(windows, keys, target_months, ages, as_of=None, stats=None):
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
    # 신차 비중은 기준 시점까지의 시장 자료로: 학습 행은 대상월 직전 달, 재귀 예측은 기준월(호출자가 넘김)
    if stats is None:
        raise ValueError('Entrant and brand statistics are required')
    as_of = months - 1 if as_of is None else pd.PeriodIndex(as_of, freq='M')
    entrants = efe.entrant_features(table['Country'], as_of, stats['entrant'])
    for name in efe.COLUMNS:
        table[name] = entrants[name].to_numpy(float)
    brands = bfe.brand_features(table['Country'], table['Brand'], as_of, stats['brand'])
    for name in bfe.COLUMNS:
        table[name] = brands[name].to_numpy(float)
    life = lfe.lifecycle_features(table['Country'], table['Brand'], table['Model'], months, as_of)
    for name in lfe.COLUMNS:
        table[name] = life[name].to_numpy(float)
    org = ofe.origin_features(table['Country'], table['Brand'], as_of, stats['origin'])
    for name in ofe.COLUMNS:
        table[name] = org[name].to_numpy(float)
    off = offe.official_features(table['Country'], as_of, stats['official'])
    for name in offe.COLUMNS:
        table[name] = off[name].to_numpy(float)
    return table[KEY + ['Month'] + [c for c in FEATURE_COLUMNS if c not in KEY]]


def build_training_data(data, cutoff):
    observed = observations(data, cutoff)
    base, audit = core.build_training_frame(observed, cutoff)
    stats = feature_stats(observed)
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
        table = make_features(windows, [key] * len(targets), targets.Month, ages, stats=stats)
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
    return np.log1p(actual) - np.log1p(anchor)


def inverse_target(raw, anchor, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown variant')
    transformed = np.asarray(raw, float)
    with np.errstate(over='ignore', invalid='ignore'):
        return np.maximum(0., np.expm1(transformed + np.log1p(np.asarray(anchor, float))))


def validate_schema(schema):
    variant = schema.get('variant')
    if (variant not in VARIANTS or schema.get('feature_columns') != FEATURE_COLUMNS
            or schema.get('feature_policy') != 'own_past_24_plus_calendar_all_signals_v1'
            or schema.get('scope') != 'global_catboost_yoy_lcg_s44'):
        raise ValueError('Invalid rich-history model schema')
    if (schema.get('calendar_definition') != cf.DEFINITION
            or schema.get('bundle_source_hashes') != _source_hashes()):
        raise ValueError('Stored calendar definition or source changed')
    expected = 'log1p_sales_minus_log1p_lag12'
    if (schema.get('capacity') != CAPACITIES[variant]
            or schema.get('objective_id') != OBJECTIVE
            or schema.get('intercept_policy') != 'mean_log_training_labels'
            or any(name in schema for name in ['asymmetry_tau', 'transition_width_log', 'loss_source_sha256'])):
        raise ValueError('Capacity or objective configuration disagrees with variant')
    parameters = schema.get('parameters')
    if parameters is not None:
        fixed = cb_settings(variant)
        fixed.pop('cat_features')
        if any(parameters.get(name) != value for name, value in fixed.items()) or parameters.get('early_stopping_rounds') is not None:
            raise ValueError('Stored parameters disagree with fixed capacity experiment')
    if (not isinstance(schema.get('base_score'), (int, float))
            or isinstance(schema['base_score'], bool) or not np.isfinite(schema['base_score'])
            or schema['base_score'] < 0):
        raise ValueError('Training-only log-sales intercept must be finite and nonnegative')
    if (schema.get('target_transform') != expected or schema.get('prediction_unit') != 'sales'
            or schema.get('prediction_inverse') != 'expm1_plus_log1p_lag12'
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


WEIGHT_RULE = 'series mean training Sales / mean over series'


def training_weights(rows):
    """학습 행마다 그 차종의 학습 행 평균 판매량 ÷ 차종 평균(차종 단위로 평균 1)."""
    means = rows.groupby(KEY, sort=False)['Sales'].transform('mean')
    series_means = rows.groupby(KEY, sort=False)['Sales'].mean()
    return (means / float(series_means.mean())).to_numpy(float)


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
    settings = {**cb_settings(variant), 'base_score': float(np.log1p(training.Sales.to_numpy(float)).mean())}
    schema = {'scope': 'global_catboost_yoy_lcg_s44', 'variant': variant, 'weight_rule': None, 'capacity': CAPACITIES[variant].copy(),
              'objective_id': OBJECTIVE,
              'calendar_definition': json.loads(json.dumps(cf.DEFINITION)),
              'bundle_source_hashes': _source_hashes(),
              'intercept_policy': 'mean_log_training_labels', 'base_score': settings['base_score'],
              'feature_columns': FEATURE_COLUMNS.copy(),
              'feature_dtypes': {c: 'category' if c in gc.CATEGORICAL_FEATURES else 'int32' if c == 'month' else 'float64' for c in FEATURE_COLUMNS},
              'feature_policy': 'own_past_24_plus_calendar_all_signals_v1',
              'target_transform': 'log1p_sales_minus_log1p_lag12',
              'prediction_inverse': 'expm1_plus_log1p_lag12',
              'prediction_unit': 'sales', 'countries': sorted(training.Country.unique().tolist()),
              'categorical_levels': {c: sorted(training[c].unique().tolist()) for c in gc.CATEGORICAL_FEATURES},
              'seen_series': training[KEY].drop_duplicates().values.tolist(), 'train_end': str(cutoff),
              'training_rows': len(training), 'training_target_start': str(training.Month.min()),
              'training_target_end': str(training.Month.max())}
    inputs = feature_frame(training, schema)
    target = transform_target(training.Sales, training.lag_12, variant)
    model = CatBoostRegressor(**{k: v for k, v in settings.items() if k != 'base_score'})
    model.fit(cb_frame(inputs), target)
    schema['parameters'] = {k: v for k, v in model.get_params().items() if k != 'cat_features'}
    schema['code_source_sha256'] = helpers.file_sha256(Path(__file__))
    schema['booster_objective_for_inference'] = OBJECTIVE
    schema['target_range'] = [float(target.min()), float(target.max())]
    canonical = training[KEY + ['Month', 'Sales'] + [c for c in FEATURE_COLUMNS if c not in KEY]].copy()
    canonical['Month'] = canonical.Month.astype(str)
    for name in ['Sales'] + [c for c in FEATURE_COLUMNS if c not in KEY]:
        canonical[name] = canonical[name].astype('int32' if name == 'month' else 'float64')
    schema['training_input_sha256'] = hashlib.sha256(pd.util.hash_pandas_object(canonical, index=False).to_numpy('uint64').tobytes()).hexdigest()
    schema['input_sales_unit'] = 'sales'
    schema['model_output_unit'] = 'log1p_sales_minus_log1p_lag12'
    model.get_metadata()[BINDING] = core._schema_digest(schema)
    return model, schema


def forecast(model, history, origin, horizon, schema):
    validate_schema(schema)
    origin = core._month(origin)
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1 or core._month(schema['train_end']) > origin:
        raise ValueError('Invalid horizon or training cutoff')
    observed = observations(history, origin, reject_future=True)
    stats = feature_stats(observed)  # 기준월 이하 모든 차종으로 계산(예측 중 고정)
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
                              [ages[i] + step for i in active], as_of=[origin] * len(active), stats=stats)
        try:
            raw = np.asarray(model.predict(cb_frame(feature_frame(batch, schema))), float)
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
    if list(model.feature_names_) != FEATURE_COLUMNS or model.get_metadata().get(BINDING) != core._schema_digest(schema):
        raise ValueError('Model/schema hash binding mismatch')
    if (schema.get('booster_objective_for_inference') != OBJECTIVE
            or schema.get('parameters', {}).get('loss_function') != OBJECTIVE
            or schema.get('code_source_sha256') != helpers.file_sha256(Path(__file__))):
        raise ValueError('Stored code or inference contract mismatch')
    if (schema.get('input_sales_unit') != 'sales' or schema.get('model_output_unit') != 'log1p_sales_minus_log1p_lag12'
            or len(schema.get('training_input_sha256', '')) != 64):
        raise ValueError('Stored training inputs or units are invalid')
    if model.tree_count_ != schema['capacity']['n_estimators']:
        raise ValueError('Actual trained capacity disagrees with schema')


def save(model, schema, path):
    path = Path(path)
    schema_path = path.with_name('feature_schema.json')
    if path.exists() or schema_path.exists():
        raise FileExistsError('Existing model artifacts cannot be overwritten')
    if path.suffix != '.cbm':
        raise ValueError('Model must use CatBoost .cbm')
    bound_model(model, schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    directory = path.parent / 'source_code'
    directory.mkdir(exist_ok=False)
    for name in schema['bundle_source_hashes']:
        shutil.copy2(Path(__file__).parent / name, directory / name)
    helpers.write_json(schema_path, schema)


def load(path):
    path = Path(path)
    schema = json.loads(path.with_name('feature_schema.json').read_text(encoding='utf-8'))
    model = CatBoostRegressor()
    model.load_model(str(path))
    model._daizen_source_code = str(path.parent / 'source_code')
    bound_model(model, schema)
    return model, schema
