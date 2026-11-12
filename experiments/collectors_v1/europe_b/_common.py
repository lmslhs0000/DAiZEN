"""europe_b 수집기 공통 도구: 예절 지키는 HTTP 세션(호스트당 2초 이상, 429면 중단), 캐시 경로.

- 호스트별 마지막 요청 시각을 기억해 최소 간격(기본 2초)을 지킨다.
- HTTP 429를 받으면 HostBlocked 예외를 내고 그 호스트는 이번 실행에서 다시 요청하지 않는다.
- robots.txt 확인 결과(2026-10-15): pfa-auto.fr, unrae.it, auto.swiss = 전부 허용(Disallow 빈칸),
  discodata.eea.europa.eu = robots.txt 없음(404), api.cepik.gov.pl = robots.txt 없음(404, API 문서만).
"""
from __future__ import annotations

import hashlib
import ssl
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
ROOT = HERE.parents[2]  # D:/daizen
RAW_DATA = ROOT / "raw" / "data.csv"
SAMPLES = ROOT / "experiments" / "data_sources_v1" / "europe_deep" / "samples"

UA = "Mozilla/5.0 (compatible; daizen-collectors/1.0; research use)"
MIN_INTERVAL = {"default": 2.0}


class HostBlocked(RuntimeError):
    """HTTP 429 등으로 이 호스트 요청을 멈춰야 할 때."""


class _LegacyTLS(HTTPAdapter):
    """api.cepik.gov.pl 은 작은 DH 키를 써서 기본 OpenSSL 보안수준에서 실패한다 → SECLEVEL=1."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


class PoliteSession:
    def __init__(self, legacy_tls_hosts: tuple[str, ...] = ("api.cepik.gov.pl",)):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        for h in legacy_tls_hosts:
            self.s.mount(f"https://{h}", _LegacyTLS())
        self.last: dict[str, float] = {}
        self.blocked: set[str] = set()
        self.n_requests: dict[str, int] = {}

    def get(self, url: str, params=None, timeout: float = 120, **kw) -> requests.Response:
        host = urlparse(url).netloc
        if host in self.blocked:
            raise HostBlocked(f"{host} 은 이번 실행에서 중단됨(이전 429)")
        gap = MIN_INTERVAL.get(host, MIN_INTERVAL["default"])
        wait = self.last.get(host, 0) + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            r = self.s.get(url, params=params, timeout=timeout, **kw)
        finally:
            self.last[host] = time.time()
            self.n_requests[host] = self.n_requests.get(host, 0) + 1
        if r.status_code == 429:
            self.blocked.add(host)
            raise HostBlocked(f"{host} HTTP 429 → 중단")
        return r


def cache_path(prefix: str, key: str, ext: str) -> Path:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return CACHE / f"{prefix}_{h}.{ext}"


def fetch_cached(sess: PoliteSession, url: str, path: Path, params=None, binary=True) -> Path:
    """캐시에 있으면 그대로, 없으면 받아서 저장."""
    if path.exists() and path.stat().st_size > 0:
        return path
    r = sess.get(url, params=params)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
