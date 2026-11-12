"""europe_c 공통 도구.

- 경로: europe_c(HERE), cache/, out/, data_sources_v1 표본(SAMPLES), europe_a·europe_b 폴더.
- 예절 지키는 HTTP: europe_b/_common.PoliteSession 을 재사용(호스트당 최소 간격, 429면 그 호스트 중단).
  여기서 호스트별 간격(robots Crawl-delay)과 **절대 요청하지 않는 주소**(FORBIDDEN)를 더한다.
- 중간 출력: out/<cc>_model_month_<YYYY-MM>.csv  (country, month, make, model, units[, 추가 열])
  + out/<cc>_meta_<YYYY-MM>.json (출처 파일, sha256, 파일 안 총계 대조 등).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # D:/daizen
CACHE = HERE / "cache"
OUT = HERE / "out"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
SAMPLES = ROOT / "experiments" / "data_sources_v1" / "europe_deep" / "samples"
EA = HERE.parent / "europe_a"
EB = HERE.parent / "europe_b"
RAW_DATA = ROOT / "raw" / "data.csv"   # 읽기만 한다
LAST_ALLOWED_TEST_MONTH = "2026-07"     # 사용자 지시: 시험·검증은 이 달 이하만

if str(EB) not in sys.path:
    sys.path.insert(0, str(EB))
import _common as eb_common  # noqa: E402  (europe_b: PoliteSession, HostBlocked, sha256)

HostBlocked = eb_common.HostBlocked
sha256 = eb_common.sha256
UA = eb_common.UA

# robots.txt Crawl-delay 또는 SPEC 최소 2초 (확인 2026-10-15)
eb_common.MIN_INTERVAL.update({
    "data.gov.ro": 10.5,             # robots: Crawl-Delay 10, Disallow /api/
    "ckan.opendata.swiss": 10.5,     # robots: Crawl-delay 10 (이 폴더에서는 쓰지 않음)
    "mobilitysweden.se": 2.5,        # robots.txt 비어 있음(전부 허용)
    "trafi2.stat.fi": 3.0,           # robots.txt 없음(404); 응답이 느림(60~70초)
    "data.public.lu": 2.5,           # robots.txt 없음(홈페이지가 돌아옴)
    "download.data.public.lu": 2.5,
    "www.auto.swiss": 2.5,           # robots: Disallow 빈칸
    "ws.cso.ie": 2.5,                # robots.txt 없음(404)
})

# 절대 요청하지 않는 곳: (호스트, 경로 접두어). 경로 ''는 그 호스트 전체.
FORBIDDEN = [
    ("api.cepik.gov.pl", ""),        # 2026-10-15 HTTP 429. dane.gov.pl 1558 의 '파일'도 이 호스트(/pliki)라서 금지
    ("dataut.vegvesen.no", "/api"),  # robots 금지 경로(이전 하위 작업이 잘못 요청함)
    ("opendata.swiss", "/api"),      # robots 금지 경로
    ("ckan.opendata.swiss", "/api"),
    ("data.gov.ro", "/api/"),        # robots 금지
]
PL_BLOCK_FILE = EB / "cache" / "pl" / "BLOCKED_429.txt"


class Forbidden(RuntimeError):
    pass


def check_allowed(url: str) -> None:
    u = urlparse(url)
    for host, pre in FORBIDDEN:
        if (u.netloc == host or u.netloc.endswith("." + host)) and u.path.startswith(pre):
            raise Forbidden(f"요청 금지 주소: {url} ({host}{pre or ' 전체'})")


class Session(eb_common.PoliteSession):
    """PoliteSession + 금지 주소 검사."""

    def get(self, url, params=None, timeout: float = 120, **kw):
        check_allowed(url)
        return super().get(url, params=params, timeout=timeout, **kw)

    def post_json(self, url, payload, timeout: float = 300):
        check_allowed(url)
        import time
        host = urlparse(url).netloc
        if host in self.blocked:
            raise HostBlocked(host)
        gap = eb_common.MIN_INTERVAL.get(host, eb_common.MIN_INTERVAL["default"])
        wait = self.last.get(host, 0) + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            r = self.s.post(url, json=payload, timeout=timeout)
        finally:
            self.last[host] = time.time()
            self.n_requests[host] = self.n_requests.get(host, 0) + 1
        if r.status_code == 429:
            self.blocked.add(host)
            raise HostBlocked(f"{host} HTTP 429 → 중단")
        return r


def check_month(m: str) -> str:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", m or ""):
        raise SystemExit(f"--month 형식 오류: {m!r} (YYYY-MM)")
    return m


def prev_month(m: str, k: int = 1) -> str:
    y, mo = int(m[:4]), int(m[5:7])
    t = y * 12 + (mo - 1) - k
    return f"{t // 12:04d}-{t % 12 + 1:02d}"


def write_model_month(cc: str, month: str, df: pd.DataFrame, meta: dict, suffix: str = "") -> Path:
    """중간 출력 저장. df 필수 열: country, month, make, model, units."""
    need = ["country", "month", "make", "model", "units"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"중간 출력 열 누락: {miss}")
    df = df[df["units"].notna()].copy()
    if (df["units"] < 0).any():
        raise ValueError("음수 대수")
    p = OUT / f"{cc.lower()}_model_month_{month}{suffix}.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    meta = {**meta, "rows": int(len(df)), "units_total": float(df["units"].sum()), "output": str(p)}
    (OUT / f"{cc.lower()}_meta_{month}{suffix}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return p


def find_cached(patterns: list[Path]) -> Path | None:
    for p in patterns:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None
