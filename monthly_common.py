"""step1_collect.py·step2_train.py 공통: 경로, Git Bash 찾기, 화면+로그 동시 출력 실행."""
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def new_log(kind):
    log = ROOT / "handover_logs" / "monthly" / f"{kind}_{datetime.now():%Y%m%d_%H%M%S}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    return log


def find_bash():
    for p in [r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files (x86)\Git\bin\bash.exe"]:
        if Path(p).exists():
            return p
    b = shutil.which("bash")
    if b and "system32" not in b.lower():          # WSL bash 말고 Git Bash
        return b
    raise SystemExit("Git Bash(bash.exe)를 찾지 못함 — Git for Windows 설치 필요")


def run(cmd, log):
    print(f"{datetime.now():%F %T} 시작: {' '.join(map(str, cmd))}\n로그: {log}\n", flush=True)
    with open(log, "a", encoding="utf-8") as f:
        # stdin=DEVNULL: 입력창 없는 실행(작업 스케줄러·숨은 창)에서도 자식·손자가 멀쩡한 입력 핸들을 받게(2026-10-30 WinError 6)
        p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
        return p.wait()
