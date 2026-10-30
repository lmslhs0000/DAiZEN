"""기록 파일에 남은 과거 절대경로를 현재 프로젝트 폴더 기준 경로로 바꾼다.

과거 실행 기록(model_manifest.json 등)은 해시 검산 대상이므로 수정하지 않고, 읽을 때만 경로를 바꾼다.
"""

import ntpath
from pathlib import Path, PureWindowsPath

PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_ROOT = PureWindowsPath(r"C:\Users\이원빈\Desktop\daizen")


def resolve_recorded_path(recorded_path, project_root=PROJECT_ROOT):
    """기록된 경로를 현재 프로젝트 폴더 안의 경로로 반환한다. 프로젝트 밖을 가리키면 중단한다."""
    root = PureWindowsPath(ntpath.normpath(str(project_root)))
    recorded = PureWindowsPath(recorded_path)
    if recorded.is_relative_to(LEGACY_ROOT):
        recorded = recorded.relative_to(LEGACY_ROOT)
    candidate = PureWindowsPath(ntpath.normpath(str(root / recorded)))
    if not candidate.is_relative_to(root):
        raise ValueError(f"프로젝트 폴더 밖의 경로는 읽지 않습니다: {recorded_path}")
    return Path(candidate)
