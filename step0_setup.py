"""0단계(처음 한 번): 이 폴더에서 수집·재학습에 필요한 파이썬 환경 7개를 만들고 준비 상태를 점검한다.

실행(이 폴더에서):  uv run step0_setup.py
  - uv 가 루트 환경(.venv: TFT·N-HiTS·XGBoost·조립)을 uv.lock 그대로 먼저 만든 뒤 이 파일을 실행한다.
  - 나머지 6개(CatBoost·ETS·TimesFM·Chronos·LightGBM·수집기)는 env/requirements_*.txt 의 고정 버전으로 만든다.
  - 이미 있는 환경은 건너뛴다(다시 만들려면 --force). 인터넷 필요(처음 약 10~30분, 패키지 약 10GB).
  - 끝에 점검표(환경·Git Bash·GPU·사전학습 모델·자료)를 찍고 setup_report.txt 로 남긴다. 모두 OK 면 step1·step2 를 쓸 수 있다.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENV = ROOT / "env"
# 이름, 환경 위치, 고정 버전 파일, torch 받는 곳(없으면 PyPI), 점검용 import
ENVS = [
    ("catboost", "experiments/catboost_v1/.venv", "requirements_catboost.txt", None, "import catboost, pandas"),
    ("ets", "experiments/ets_v1/.venv", "requirements_ets.txt", None, "import statsmodels, pandas"),
    ("timesfm", "experiments/timesfm_v1/.venv", "requirements_timesfm.txt", "https://download.pytorch.org/whl/cpu", "import timesfm, torch"),
    ("chronos_ft", "experiments/chronos_ft_v1/.venv", "requirements_chronos_ft.txt", "https://download.pytorch.org/whl/cu128", "import chronos, torch"),
    ("lightgbm", "experiments/lightgbm_v1/.venv", "requirements_lightgbm.txt", None, "import lightgbm, pandas"),
    ("collect", "experiments/collectors_v1/.venv", "requirements_collect.txt", None, "import requests, bs4, lxml, openpyxl, xlrd, pdfplumber, pandas"),
]
MODELS = ["models--amazon--chronos-2", "models--google--timesfm-2.5-200m-pytorch"]


def sh(cmd, **kw):
    print("  $ " + " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=ROOT, **kw).returncode


def py_of(rel):
    return ROOT / rel / "Scripts" / "python.exe"


def make_env(name, rel, req, torch_index, force):
    py = py_of(rel)
    if py.exists() and not force:
        print(f"[{name}] 이미 있음 → 건너뜀")
        return
    if force and (ROOT / rel).exists():
        shutil.rmtree(ROOT / rel)
    print(f"[{name}] 만드는 중")
    (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)   # v6: 실행 파일만 담은 묶음에는 빈 실험 폴더가 없을 수 있다
    if sh(["uv", "venv", str(ROOT / rel), "--python", "3.12"]) != 0:
        raise SystemExit(f"{name}: uv venv 실패")
    cmd = ["uv", "pip", "install", "--python", str(py), "-r", str(ENV / req)]
    if torch_index:
        cmd += ["--extra-index-url", torch_index, "--index-strategy", "unsafe-best-match"]
    if sh(cmd) != 0:
        raise SystemExit(f"{name}: 패키지 설치 실패")


def check(name, py, code):
    if not Path(py).exists():
        return "없음"
    r = subprocess.run([str(py), "-c", code], cwd=ROOT, capture_output=True, text=True)
    return "OK" if r.returncode == 0 else "import 실패: " + (r.stderr.strip().splitlines() or ["?"])[-1][:120]


def find_bash():
    for p in [r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"]:
        if Path(p).exists():
            return p
    b = shutil.which("bash")
    return b if b and "system32" not in b.lower() else None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="이미 있는 환경도 지우고 다시 만듦")
    ap.add_argument("--check", action="store_true", help="만들지 않고 점검만")
    a = ap.parse_args()
    if not shutil.which("uv"):
        raise SystemExit("uv 가 없음 — https://docs.astral.sh/uv/ 에서 설치한 뒤 다시 실행")
    if not a.check:
        print("[root] uv sync --frozen")
        if sh(["uv", "sync", "--frozen"]) != 0:
            raise SystemExit("root: uv sync 실패")
        for name, rel, req, idx, _ in ENVS:
            make_env(name, rel, req, idx, a.force)
    rows = [("환경 root(.venv)", check("root", py_of(".venv"), "import torch, xgboost, pytorch_forecasting, pandas"))]
    rows += [(f"환경 {n}({rel})", check(n, py_of(rel), code)) for n, rel, _, _, code in ENVS]
    bash = find_bash()
    rows.append(("Git Bash(재학습에 필요)", f"OK {bash}" if bash else "없음 — Git for Windows 설치 필요"))
    gpu = subprocess.run([str(py_of("experiments/chronos_ft_v1/.venv")), "-c", "import torch;print(torch.cuda.is_available())"],
                         capture_output=True, text=True).stdout.strip() if py_of("experiments/chronos_ft_v1/.venv").exists() else "?"
    rows.append(("GPU(CUDA)", "OK" if gpu == "True" else "없음 — Chronos-2 미세조정을 CPU 로 함(재학습이 몇 시간 더 걸림)"))
    hub = ROOT / ".cache" / "huggingface" / "hub"
    if not a.check:                       # GitHub 에서 받은 사본엔 모델이 없다(100MB 제한) → 설치 때 내려받음
        for m in MODELS:
            if not (hub / m).exists():
                rid = m.replace("models--", "").replace("--", "/")
                print(f"[모델] {rid} 내려받는 중")
                subprocess.run([str(py_of("experiments/timesfm_v1/.venv")), "-c",
                                f"import os;os.environ['HF_HOME']=r'{ROOT / '.cache' / 'huggingface'}';"
                                f"from huggingface_hub import snapshot_download;snapshot_download('{rid}')"], cwd=ROOT)
    rows += [(f"사전학습 모델 {m.split('--', 2)[-1]}", "OK" if (hub / m).exists() else "없음 — 인터넷 연결 뒤 다시 실행") for m in MODELS]
    rows.append(("자료 raw/data.csv", "OK" if (ROOT / "raw" / "data.csv").exists() else "없음"))
    w = max(len(r[0]) for r in rows)
    text = "\n".join(f"{k.ljust(w)}  {v}" for k, v in rows)
    print("\n점검 결과\n" + text)
    (ROOT / "setup_report.txt").write_text(text + "\n", encoding="utf-8")
    bad = [k for k, v in rows if not v.startswith("OK") and not k.startswith(("GPU", "사전학습"))]
    print("\n" + ("준비 완료 → step1_collect.py(수집), step2_train.py(재학습)" if not bad else f"고칠 것: {', '.join(bad)}"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
