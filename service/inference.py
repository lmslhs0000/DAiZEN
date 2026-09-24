import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import io
import json
import logging
import math
from pathlib import Path
import re
import threading
import uuid

import pandas as pd

from baseline import SERIES_KEY, read_sales_csv
from service.errors import RequestError, ServiceError
from service.registry import ModelRegistry

LOGGER = logging.getLogger(__name__)
MAX_CSV_BYTES = 25 * 1024 * 1024
FORECAST_FIELDS = ['Country', 'Brand', 'Model', 'Forecast Month', 'Horizon', 'Predicted Sales']
EXCLUDED_FIELDS = ['Country', 'Brand', 'Model', 'status', 'reason']


def _read_input(source):
    try:
        if isinstance(source, bytes):
            content = source
        else:
            path = Path(source)
            if path.stat().st_size > MAX_CSV_BYTES:
                raise RequestError('CSV_TOO_LARGE', 'CSV 최대 크기는 25 MiB입니다.')
            content = path.read_bytes()
        if len(content) > MAX_CSV_BYTES:
            raise RequestError('CSV_TOO_LARGE', 'CSV 최대 크기는 25 MiB입니다.')
        text = content.decode('utf-8-sig')
        if '\x00' in text:
            raise ValueError('NUL byte is not valid CSV input')


        reader = csv.reader(io.StringIO(text, newline=''), strict=True)
        header = next(reader)
        if not header or len(set(header)) != len(header) or any(not name.strip() for name in header):
            raise ValueError('CSV header names must be nonblank and unique')
        for line, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f'CSV record {line} has a different number of fields')
        return read_sales_csv(io.BytesIO(content))
    except RequestError:
        raise
    except (ValueError, OSError, UnicodeError, csv.Error, StopIteration) as error:

        message = 'CSV를 읽을 수 없습니다.' if isinstance(error, OSError) else str(error).split('\n', 1)[0][:350]
        raise RequestError('INVALID_CSV', message or 'CSV에 헤더와 판매 데이터가 필요합니다.') from error


def _origin(value, maximum):
    if value is None:
        return maximum
    if not isinstance(value, str) or re.fullmatch(r'\d{4}-\d{2}', value) is None:
        raise RequestError('INVALID_ORIGIN', 'origin은 YYYY-MM 형식이어야 합니다.')
    try:
        month = pd.Period(value, freq='M')
    except ValueError as error:
        raise RequestError('INVALID_ORIGIN', '유효한 origin 월이 필요합니다.') from error
    if month > maximum:
        raise RequestError('ORIGIN_AFTER_DATA', 'origin은 CSV의 마지막 월보다 뒤일 수 없습니다.')
    return month


def _format_market(predictions, statuses, expected_keys, origin, horizon):

    try:
        if statuses.duplicated(SERIES_KEY).any():
            raise ValueError('Duplicate status')
        status_by_key = {tuple(row[col] for col in SERIES_KEY): row for row in statuses.to_dict('records')}
        if set(status_by_key) != expected_keys:
            raise ValueError('Missing or unexpected status keys')
        groups = {key: rows.sort_values('horizon') for key, rows in predictions.groupby(SERIES_KEY, sort=True)}
        if set(groups) - expected_keys:
            raise ValueError('Unexpected prediction keys')
        forecasts, excluded = [], []
        for key in sorted(expected_keys):
            identity = dict(zip(SERIES_KEY, key))
            state = status_by_key[key]
            status, reason = state['prediction_status'], state['reason']
            if status == 'OK':
                rows = groups.get(key)
                valid = rows is not None and len(rows) == horizon
                if valid:
                    valid = (rows['horizon'].tolist() == list(range(1, horizon + 1))
                             and rows['Month'].tolist() == [origin+i for i in range(1, horizon+1)]
                             and all(math.isfinite(float(v)) and float(v) >= 0 for v in rows['prediction']))
                if not valid:
                    status, reason = 'PREDICTION_ERROR', '요청한 전체 기간의 유효한 예측을 만들지 못했습니다.'
                else:
                    for row in rows.to_dict('records'):
                        forecasts.append({**identity, 'Forecast Month': str(row['Month']),
                                          'Horizon': int(row['horizon']), 'Predicted Sales': float(row['prediction'])})
            if status != 'OK':
                if status == 'PREDICTION_ERROR':
                    LOGGER.error('Forecast failure for %s: %s', key, reason)
                    reason = '전체 기간 예측에 실패했습니다. 서버 로그에서 원인을 확인할 수 있습니다.'
                excluded.append({**identity, 'status': str(status), 'reason': str(reason)})
        return forecasts, excluded
    except Exception as error:
        LOGGER.exception('Invalid forecast/status contract')
        raise ServiceError('INVALID_MODEL_OUTPUT', '모델 결과의 시계열 또는 상태 구성이 올바르지 않습니다.') from error


class Predictor:
    def __init__(self, config_path=None, *, project_root=None, registry=None):
        self.registry = registry if registry is not None else ModelRegistry(config_path, project_root=project_root)
        self._lock = threading.Lock()

    def predict_csv(self, source, *, origin=None, horizon_months=24):
        result = self.predict_csv_detailed(source, origin=origin, horizon_months=horizon_months)
        return {
            'predictions': [
                {**row, 'Predicted Sales': int(Decimal(str(row['Predicted Sales'])).to_integral_value(rounding=ROUND_HALF_UP))}
                for row in result['predictions']
            ],
            'excluded': result['excluded'],
        }

    def predict_csv_detailed(self, source, *, origin=None, horizon_months=24):
        if type(horizon_months) is not int or not 1 <= horizon_months <= 24:
            raise RequestError('INVALID_HORIZON', 'horizon_months는 1~24의 정수여야 합니다.')
        data = _read_input(source)
        cutoff = _origin(origin, data['Month'].max())
        universe = data[SERIES_KEY].drop_duplicates().sort_values(SERIES_KEY)
        countries = sorted(universe['Country'].unique())
        models = [self.registry.metadata(country) for country in countries if country in self.registry.entries]
        if any(pd.Period(model['train_end'], freq='M') > cutoff for model in models):
            raise RequestError('ORIGIN_BEFORE_MODEL_TRAINING', 'origin은 선택된 모델의 학습 종료월(2026-07) 이후 또는 같은 월이어야 합니다.')
        history = data.loc[data['Month'].le(cutoff)].copy()
        forecasts, excluded = [], []


        with self._lock:
            for country in countries:
                keys = {tuple(row) for row in universe.loc[universe['Country'].eq(country)].values.tolist()}
                if country not in self.registry.entries:
                    excluded.extend({**dict(zip(SERIES_KEY, key)), 'status': 'UNSUPPORTED_COUNTRY',
                                     'reason': '서버 설정에 해당 Country의 모델이 없습니다.'} for key in sorted(keys))
                    continue
                country_history = history.loc[history['Country'].eq(country)].copy()
                available = {tuple(row) for row in country_history[SERIES_KEY].drop_duplicates().values.tolist()}
                excluded.extend({**dict(zip(SERIES_KEY, key)), 'status': 'INSUFFICIENT_HISTORY',
                                 'reason': 'origin 이하에 관측 이력이 없습니다.'} for key in sorted(keys-available))
                if not available:
                    continue
                try:
                    predictions, statuses = self.registry.forecast(country, country_history, cutoff, horizon_months)
                except ServiceError:
                    raise
                except Exception as error:
                    LOGGER.exception('Unexpected core inference failure for %s', country)
                    raise ServiceError('INFERENCE_FAILED', f'{country} 모델 실행 중 오류가 발생했습니다.') from error
                success, failures = _format_market(predictions, statuses, available, cutoff, horizon_months)
                forecasts.extend(success)
                excluded.extend(failures)
        forecasts.sort(key=lambda r: tuple(r[k] for k in SERIES_KEY)+ (r['Horizon'],))
        excluded.sort(key=lambda r: tuple(r[k] for k in SERIES_KEY))
        predicted_keys = {tuple(row[k] for k in SERIES_KEY) for row in forecasts}
        excluded_keys = {tuple(row[k] for k in SERIES_KEY) for row in excluded}
        if predicted_keys & excluded_keys or len(predicted_keys)+len(excluded_keys) != len(universe):
            raise ServiceError('INVALID_MODEL_OUTPUT', '입력 시계열과 예측·제외 결과가 일치하지 않습니다.')
        warnings = []
        ignored = len(data)-len(history)
        if ignored:
            warnings.append({'code': 'FUTURE_ROWS_IGNORED', 'message': f'origin 이후 {ignored}행은 예측 이력에서 제외했습니다.'})
        if excluded:
            warnings.append({'code': 'SERIES_EXCLUDED', 'message': f'{len(excluded)}개 시계열은 excluded에서 사유를 확인하세요.'})
        return {
            'request_id': uuid.uuid4().hex, 'origin': str(cutoff), 'horizon_months': horizon_months,
            'models': models,
            'summary': {'input_rows': len(data), 'history_rows': len(history), 'ignored_future_rows': ignored,
                        'input_series': len(universe), 'forecasted_series': len(predicted_keys),
                        'excluded_series': len(excluded_keys), 'forecast_rows': len(forecasts)},
            'predictions': forecasts, 'excluded': excluded, 'warnings': warnings,
        }


def output_paths(output):
    output = Path(output)
    return output, output.with_name('forecasft_data.csv'), output.with_name('excluded_vehicles.csv')


def save_outputs(result, output):
    json_path, forecast_path, excluded_path = output_paths(output)
    for path in (json_path, forecast_path, excluded_path):
        if path.exists():
            raise FileExistsError(f'기존 결과를 덮어쓸 수 없습니다: {path}')
    if len({path.resolve() for path in (json_path, forecast_path, excluded_path)}) != 3:
        raise ValueError('JSON과 CSV의 저장 파일명은 달라야 합니다.')
    with json_path.open('x', encoding='utf-8') as destination:
        json.dump(result, destination, ensure_ascii=False, allow_nan=False, indent=2)
    for path, fields, rows in ((forecast_path, FORECAST_FIELDS, result['predictions']),
                               (excluded_path, EXCLUDED_FIELDS, result['excluded'])):
        with path.open('x', encoding='utf-8-sig', newline='') as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main(argv=None):
    from service.console import configure_console
    configure_console()
    parser = argparse.ArgumentParser(description='학습 없이 저장된 모델로 1~24개월 예측')
    parser.add_argument('--csv', required=True)
    parser.add_argument('--config')
    parser.add_argument('--origin')
    parser.add_argument('--horizon-months', type=int, default=24)
    parser.add_argument('--output', required=True, help='새 JSON 파일; 같은 폴더에 예측·제외 CSV도 저장')
    args = parser.parse_args(argv)
    paths = output_paths(args.output)
    if len({path.resolve() for path in paths}) != 3:
        parser.error('JSON과 CSV의 저장 파일명은 달라야 합니다.')
    if any(path.exists() for path in paths):
        parser.error('기존 출력 파일을 덮어쓸 수 없습니다.')
    try:
        result = Predictor(args.config).predict_csv(args.csv, origin=args.origin, horizon_months=args.horizon_months)
        save_outputs(result, args.output)
    except (RequestError, ServiceError) as error:
        parser.exit(2, f'{error.code}: {error.message}\n')
    except OSError as error:
        parser.exit(2, f'결과 저장 실패: {error}\n')
    print(f"예측 {len(result['predictions']):,}행 / 제외 차량 {len(result['excluded']):,}개")


if __name__ == '__main__':
    main()
