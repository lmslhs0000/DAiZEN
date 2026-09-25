import hashlib
import json
import logging
from pathlib import Path, PureWindowsPath
import re

from service.errors import ServiceError

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / 'configs' / 'tft_a.json'
ORIGINAL_RUNS = {
    'XGBoost': 'run_20260921T175602_851343Z_28c46a24e7d2',
    'TFT': 'run_20260922T081432_215155Z_3a3e74e049cd',
}


APPROVED_REFRESH_RUNS = {
    'run_20260924T190014_42dc7a14': {
        'TFT': '8df79eb2f4a1acb7499a52bbe4dd9ddcaa4892b71f4bb6d4516702fb2a56e4eb',
        'XGBoost': '041dac07d75ed0f668816f9f395a31f5878e937775decef484092c79e1357182',
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact path escapes project root')
    return path


def _configured_path(root, relative):
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError('Configured path must be root-relative')
    path = PureWindowsPath(relative)
    if path.drive or path.root or not path.parts or '..' in path.parts or any(':' in part for part in path.parts):
        raise ValueError('Configured path must be root-relative')
    return _inside(root, Path(*path.parts))


def _relocated_artifact(root, saved_path, run_id, *, algorithm=None, method=None):


    if not isinstance(saved_path, str) or not saved_path:
        raise ValueError('Invalid saved model path')
    parts = PureWindowsPath(saved_path).parts
    positions = [i for i, value in enumerate(parts) if value == 'models']
    if len(positions) != 1 or '..' in parts:
        raise ValueError('Invalid historical model path')
    suffix = parts[positions[0]:]
    if any(':' in part for part in suffix):
        raise ValueError('Invalid saved model path')
    if run_id in APPROVED_REFRESH_RUNS:
        if algorithm not in ('TFT', 'XGBoost') or len(suffix) != 9:
            raise ValueError('Invalid approved refresh artifact path')
        if suffix[:4] != ('models', 'model_refresh', run_id, algorithm.lower()):
            raise ValueError('Artifact belongs to a different refresh run or algorithm')
        attempt = suffix[7] if algorithm == 'TFT' else suffix[4]
        phase_index = 4 if algorithm == 'TFT' else 5
        if not re.fullmatch(r'attempt_[0-9a-f]+', attempt):
            raise ValueError('Invalid production attempt path')
    else:
        phase_index = 2
        if run_id not in ORIGINAL_RUNS.values() or len(suffix) < 5 or suffix[1] != run_id:
            raise ValueError('Only original production_refit artifacts are allowed')
    if suffix[phase_index] != 'production_refit':
        raise ValueError('Only production_refit artifacts are allowed')
    if method is not None and suffix[phase_index + 1] != method:
        raise ValueError('Artifact selection method mismatch')
    return _inside(root, Path(*suffix))


class ModelRegistry:
    def __init__(self, config_path=None, *, project_root=None):
        self.project_root = Path(project_root or PROJECT_ROOT).resolve()
        self.config_path = Path(config_path or DEFAULT_CONFIG).resolve()
        self.entries = {}
        self._cache = {}
        try:
            config = json.loads(self.config_path.read_text(encoding='utf-8-sig'))
            if config['version'] not in (1, 2, 3) or not isinstance(config['countries'], dict) or not config['countries']:
                raise ValueError('Invalid server configuration')
            self.name = config['name']
            sources = {}
            for algorithm in {value['algorithm'] for value in config['countries'].values()}:
                source = config['sources'][algorithm]
                run_id = source['run_id']
                if config['version'] == 3:
                    approved_hash = APPROVED_REFRESH_RUNS.get(run_id, {}).get(algorithm)
                    if not approved_hash or source['manifest_sha256'] != approved_hash:
                        raise ValueError('Refresh manifest is not explicitly approved')
                elif run_id != ORIGINAL_RUNS[algorithm]:
                    raise ValueError('Experimental/evaluation runs cannot replace originals')
                if config['version'] in (2, 3):
                    manifest = _configured_path(self.project_root, source['manifest_path'])
                else:
                    manifest = _inside(self.project_root, Path('reports') / algorithm.lower() / run_id / 'model_manifest.json')
                if sha256(manifest) != source['manifest_sha256']:
                    raise ValueError('Approved model manifest hash mismatch')
                sources[algorithm] = json.loads(manifest.read_text(encoding='utf-8'))
            for country, choice in config['countries'].items():
                if not isinstance(country, str) or not country.strip():
                    raise ValueError('Country configuration is blank')
                algorithm, method = choice['algorithm'], choice['selection_method']
                if method not in ('A', 'B'):
                    raise ValueError('Invalid selection method')
                records = [r for r in sources[algorithm] if r['Country'] == country
                           and r['phase'] == 'production_refit' and r['method'] == method]
                if len(records) != 1:
                    raise ValueError('Exactly one production bundle per country is required')
                bundle_dir = None
                if config['version'] == 2 or 'bundle_dir' in choice:
                    bundle_dir = _configured_path(self.project_root, choice['bundle_dir'])
                self.entries[country] = self._entry(
                    country, algorithm, method, records[0], bundle_dir,
                    registry_run_id=config['sources'][algorithm]['run_id'])
        except Exception as error:
            LOGGER.exception('Cannot prepare model registry')
            raise ServiceError('MODEL_UNAVAILABLE', '서버의 저장 모델 설정 또는 파일을 확인해 주세요.') from error

    def _entry(self, country, algorithm, method, record, bundle_dir=None, *, registry_run_id=None):
        registry_run_id = registry_run_id or ORIGINAL_RUNS[algorithm]
        run_id = registry_run_id
        if registry_run_id in APPROVED_REFRESH_RUNS:
            action = record['refresh_action']
            expected_action = 'refitted' if country in ('China', 'Europe', 'South Korea') else 'reused'
            expected_run = registry_run_id if action == 'refitted' else ORIGINAL_RUNS[algorithm]
            if action != expected_action or record['source_run_id'] != expected_run:
                raise ValueError('Refresh provenance does not match the approved country action')
            run_id = expected_run
        if record['phase'] != 'production_refit' or record['Country'] != country or record['method'] != method:
            raise ValueError('Production bundle identity mismatch')

        def artifact_path(saved_path):
            return _relocated_artifact(self.project_root, saved_path, run_id, algorithm=algorithm, method=method)

        path = artifact_path(record['model_path'])
        expected_name = 'model.ckpt' if algorithm == 'TFT' else 'model.ubj'
        if path.name != expected_name:
            raise ValueError('Unexpected production model filename')
        schema_path = artifact_path(record['feature_schema_path'])
        if schema_path != path.with_name('feature_schema.json'):
            raise ValueError('Schema belongs to a different bundle')
        if algorithm == 'XGBoost':
            hashes = {path: record['model_sha256'], schema_path: record['feature_schema_sha256']}
        else:
            hashes = {artifact_path(name): digest
                      for name, digest in record['artifact_hashes'].items()
                      if PureWindowsPath(name).name in (path.name, 'feature_schema.json', 'dataset_parameters.pt', 'artifact_manifest.json')}
            required = {path, schema_path, path.with_name('dataset_parameters.pt'), path.with_name('artifact_manifest.json')}
            if set(hashes) != required:
                raise ValueError('TFT bundle is incomplete')
        if bundle_dir is not None:
            hashes = {_inside(bundle_dir, artifact.name): digest for artifact, digest in hashes.items()}
            path = _inside(bundle_dir, path.name)
            schema_path = _inside(bundle_dir, schema_path.name)
        self._verify(hashes)
        schema = json.loads(schema_path.read_text(encoding='utf-8'))
        if schema['country'] != country or schema['train_end'] != record['train_end'] or record['train_end'] != '2026-07':
            raise ValueError('Country or production training cutoff mismatch')
        if algorithm == 'TFT' and (schema['history_length'], schema['prediction_length']) != (12, 1):
            raise ValueError('Original TFT input/output lengths changed')
        metadata = {
            'Country': country, 'algorithm': algorithm, 'selection_method': method,
            'bundle_id': f'{run_id}/{method}/{country}', 'train_end': schema['train_end'],
            'history_length': 12, 'prediction_length': 1, 'model_sha256': hashes[path],
        }
        if registry_run_id in APPROVED_REFRESH_RUNS:
            metadata.update(registry_run_id=registry_run_id, source_run_id=run_id,
                            refresh_action=record['refresh_action'])
        return {'path': path, 'hashes': hashes, 'schema': schema, 'metadata': metadata}

    @staticmethod
    def _verify(hashes):
        for path, expected in hashes.items():
            if sha256(path) != expected:
                raise ValueError('Saved artifact hash mismatch')

    def metadata(self, country):
        entry = self.entries[country]
        try:


            self._verify(entry['hashes'])
        except Exception as error:
            LOGGER.exception('Configured artifacts became unavailable for %s', country)
            raise ServiceError('MODEL_UNAVAILABLE', f'{country}의 저장 모델을 사용할 수 없습니다.') from error
        return dict(entry['metadata'])

    def load(self, country):
        entry = self.entries[country]
        try:
            self._verify(entry['hashes'])
            if country not in self._cache:
                if entry['metadata']['algorithm'] == 'TFT':
                    import torch
                    from tft_core import load_market_model
                    torch.set_num_threads(1)
                    bundle = load_market_model(entry['path'], device='cpu')
                    if bundle['schema'] != entry['schema']:
                        raise ValueError('Loaded TFT schema mismatch')
                else:
                    from xgboost_core import load_market_model
                    bundle = load_market_model(entry['path'])
                    if bundle[1] != entry['schema']:
                        raise ValueError('Loaded XGBoost schema mismatch')
                self._cache[country] = bundle
            return self._cache[country]
        except Exception as error:
            LOGGER.exception('Cannot load configured model for %s', country)
            raise ServiceError('MODEL_UNAVAILABLE', f'{country}의 저장 모델을 사용할 수 없습니다.') from error

    def forecast(self, country, history, origin, horizon):
        bundle = self.load(country)
        if self.entries[country]['metadata']['algorithm'] == 'TFT':
            from tft_core import forecast_recursive
            return forecast_recursive(bundle, history, origin, horizon)
        from xgboost_core import forecast_recursive
        return forecast_recursive(bundle[0], history, origin, horizon, bundle[1])
