"""유럽 한 달치: 나라별 수집기(europe_a·europe_b) → 독일 출력 연결 → europe_c 계산기(--collect).

사용: python europe_chain.py --month 2026-08 [--offline] [--germany-file <collect_germany 출력>]
europe_a·europe_b 수집기는 요청한 달만 남기고 출력 파일을 덮어쓰므로 최근 6개월을 함께 다시 만든다(검증용 이력 보존).
끝에 표준 출력 경로 한 줄(europe_c/out/std_europe_<M>.csv)을 찍는다.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

COLL = Path(__file__).resolve().parent.parent
KEEP = 6


def steps(month, offline):
    first = str(pd.Period(month, freq="M") - (KEEP - 1))
    off = ["--offline"] if offline else []
    return [
        (COLL / "europe_a", ["collect_es.py", "--origin", month, "--months", str(KEEP)]),
        (COLL / "europe_a", ["collect_nl.py", "--origin", month, "--months", str(KEEP)]),
        (COLL / "europe_a", ["collect_uk.py", "--origin", month, "--months", str(KEEP)]),
        (COLL / "europe_b", ["collect_fr.py", "--range", first, month]),
        (COLL / "europe_b", ["collect_it.py", "--range", first, month]),
        (COLL / "europe_c", ["collect_europe.py", "--month", month, "--collect"] + off),
    ]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True)
    ap.add_argument("--offline", action="store_true", help="europe_a·b 는 건너뛰고(이미 있는 출력 사용) 계산기만 캐시로")
    ap.add_argument("--germany-file")
    a = ap.parse_args()
    if a.germany_file and Path(a.germany_file).exists():   # 계산기는 rules_jp_na_de/out/germany_<M>_<M>.csv 를 읽는다
        dst = COLL / "rules_jp_na_de" / "out" / f"germany_{a.month}_{a.month}.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)      # 새로 받은 묶음에는 out 폴더가 없다(git 은 빈 폴더를 안 담음)
        shutil.copy2(a.germany_file, dst)
    failed = []
    for cwd, args in steps(a.month, a.offline):
        if a.offline and cwd.name in ("europe_a", "europe_b"):
            continue
        r = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(f"[{cwd.name} {args[0]}] code {r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}", file=sys.stderr)
        if r.returncode != 0:
            failed.append(args[0])
    out = COLL / "europe_c" / "out" / f"std_europe_{a.month}.csv"
    if not out.exists():
        raise SystemExit(f"유럽 표준 출력 없음 (실패: {failed})")
    print(out)


if __name__ == "__main__":
    main()
