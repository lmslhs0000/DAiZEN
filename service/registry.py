import hashlib
import json
import logging
from pathlib import Path, PureWindowsPath

from service.errors import ServiceError

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / 'configs' / 'tft_a.json'
ORIGINAL_RUNS = {
    'XGBoost': 'run_20260921T175602_851343Z_28c46a24e7d2',
    'TFT': 'run_20260922T081432_215155Z_3a3e74e049cd',
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


def _relocated_artifact(root, saved_path, run_id):


    parts = PureWindowsPath(saved_path).parts
    positions = [i for i, value in enumerate(parts) if value == 'models']
    if len(positions) != 1:
        raise ValueError('Invalid historical model path')
    suffix = parts[positions[0]:]
    if len(suffix) < 5 or suffix[1:3] != (run_id, 'production_refit') or '..' in suffix:
        raise ValueError('Only original production_refit artifacts are allowed')
    return _inside(root, Path(*suffix))


class ModelRegistry:
    def __init__(self, config_path=None, *, project_root=None):
        self.project_root = Path(project_root or PROJECT_ROOT).resolve()
        self.config_path = Path(config_path or DEFAULT_CONFIG).resolve()
        self.entries = {}
        self._cache = {}
        try:
            config = json.loads(self.config_path.read_text(encoding='utf-8-sig'))
            if config['version'] not in (1, 2) or not isinstance(config['countries'], dict) or not config['countries']:
                raise ValueError('Invalid server configuration')
            self.name = config['name']
            sources = {}
            for algorithm in {value['algorithm'] for value in config['countries'].values()}:
                source = config['sources'][algorithm]
                run_id = ORIGINAL_RUNS[algorithm]
                if source['run_id'] != run_id:
                    raise ValueError('Experimental/evaluation runs cannot replace originals')
                if config['version'] == 2:
                    manifest = _configured_path(self.project_root, source['manifest_path'])
                else:
                    manifest = _inside(self.project_root, Path('reports') / algorithm.lower() / run_id / 'model_manifest.json')
                if sha256(manifest) != source['manifest_sha256']:
                    raise ValueError('Original model manifest hash mismatch')
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
                bundle_dir = _configured_path(self.project_root, choice['bundle_dir']) if config['version'] == 2 else None
                self.entries[country] = self._entry(country, algorithm, method, records[0], bundle_dir)
        except Exception as error:
            LOGGER.exception('Cannot prepare model registry')
            raise ServiceError('MODEL_UNAVAILABLE', '서버의 저장 모델 설정 또는 파일을 확인해 주세요.') from error

    def _entry(self, country, algorithm, method, record, bundle_dir=None):
        run_id = ORIGINAL_RUNS[algorithm]
        path = _relocated_artifact(self.project_root, record['model_path'], run_id)
        schema_path = _relocated_artifact(self.project_root, record['feature_schema_path'], run_id)
        if schema_path != path.with_name('feature_schema.json'):
            raise ValueError('Schema belongs to a different bundle')
        if algorithm == 'XGBoost':
            hashes = {path: record['model_sha256'], schema_path: record['feature_schema_sha256']}
        else:
            hashes = {_relocated_artifact(self.project_root, name, run_id): digest
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
        return {'path': path, 'hashes': hashes, 'schema': schema, 'metadata': {
            'Country': country, 'algorithm': algorithm, 'selection_method': method,
            'bundle_id': f'{run_id}/{method}/{country}', 'train_end': schema['train_end'],
            'history_length': 12, 'prediction_length': 1, 'model_sha256': hashes[path],
        }}

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
