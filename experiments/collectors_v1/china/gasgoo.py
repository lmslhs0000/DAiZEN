"""Gasgoo (CPCA) model-page fetcher/parser shared by collect_china.py and the work/ scripts.
Site rules: public /autosales/ pages only (robots.txt allows them; /api/ is not used), >= 2.5 s between live requests,
HTTP 429 -> stop (SystemExit 2), every page cached; existing caches from earlier rounds are read first."""
import re, sys, time, html, urllib.request, urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
ROOT = HERE.parents[2]   # 프로젝트 최상위(어느 컴퓨터든 상대 경로)
READ_CACHES = [CACHE, ROOT / "experiments" / "data_sources_v1" / "group_0" / "cache_pages"]   # 2026-10-19: 개인 PC 임시 폴더 경로 제거
BASE = "https://autonews.gasgoo.com"
DELAY = 2.5
MON = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
_last = [0.0]
STATS = {"live": 0, "cache": 0, "err": 0}


def _cached(fname):
    for c in READ_CACHES:
        f = c / fname
        if f.exists() and f.stat().st_size > 1000:
            return f
    return None


def get(url, fname, refresh=False):
    """-> (html, status). status: cache | live | HTTP<code> | <ExceptionName>"""
    f = None if refresh else _cached(fname)
    if f:
        STATS["cache"] += 1
        return f.read_text(encoding="utf-8", errors="ignore"), "cache"
    w = DELAY - (time.time() - _last[0])
    if w > 0:
        time.sleep(w)
    _last[0] = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            h = r.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        STATS["err"] += 1
        if e.code == 429:
            print("HTTP 429 - stopping (site rule)", url, flush=True)
            raise SystemExit(2)
        return "", f"HTTP{e.code}"
    except Exception as e:
        STATS["err"] += 1
        return "", type(e).__name__
    STATS["live"] += 1
    if len(h) > 1000:
        (CACHE / fname).write_text(h, encoding="utf-8")
    return h, "live"


def text(h):
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", h, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", "|", t))
    return re.sub(r"\s*\|[\s|]*", "|", t)


HDR = re.compile(r"\|(\w{3})\|-\|(\d{4})\|(\w{3})\|-\|(\d{4})\|(\w{3})\|-\|(\d{4})\|(\d{4})\|Total\|")
ROW = re.compile(r"([^|]+)\|([\d,\-]+)\|([\d,\-]+)\|([\d,\-]+)\|([\d,\-]+)\|")
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())


def parse(h):
    """Model page -> dict(title, periods[3], line=(name,[v1,v2,v3],ytd) or None, oem, status).
    The page table lists the line's own row only when the line has a value in the 3-month window; otherwise the first
    row is the OEM (trap). Line row = first row if >= 4 rows (line, OEM, segment, body) or its name equals the title."""
    tt = re.search(r"<title>(.*?)</title>", h, re.S)
    title = html.unescape(tt.group(1)).split("Sales of", 1)[1].split(" - Gasgoo")[0].strip() if tt and "Sales of" in tt.group(1) else ""
    t = text(h)
    mm = HDR.search(t)
    if not mm:
        return dict(title=title, periods=[], line=None, oem="", status="page_no_table")
    g = mm.groups()
    periods = [f"{g[2*k+1]}-{MON[g[2*k]]:02d}" for k in range(3)]
    rest = t[mm.end() - 1:].split("|Welcome to browse")[0] + "|"
    rows = []
    for r in ROW.finditer(rest):
        iv = lambda x: int(x.replace(",", "")) if x.replace(",", "").isdigit() else None
        rows.append((r.group(1).strip("|").strip(), [iv(r.group(k)) for k in (2, 3, 4)], iv(r.group(5))))
    if not rows:
        return dict(title=title, periods=periods, line=None, oem="", status="page_no_rows")
    if len(rows) >= 4 or norm(rows[0][0]) == norm(title):
        return dict(title=title, periods=periods, line=rows[0], oem=rows[1][0] if len(rows) > 1 else "", status="ok")
    return dict(title=title, periods=periods, line=None, oem=rows[0][0], status="line_no_value_3m")


def model_page(slug, y, m, refresh=False):
    url = f"{BASE}/autosales/model/{slug}/{y}/{m}.html"
    h, st = get(url, f"m_{slug}_{y}_{m}.html", refresh=refresh)
    p = parse(h) if h else dict(title="", periods=[], line=None, oem="", status="fetch_" + st)
    p["url"], p["fetch"] = url, st
    return p


def page_values(p):
    """-> {period: units or None('-')} for a parsed page"""
    out = {}
    for k, per in enumerate(p.get("periods", [])):
        out[per] = p["line"][1][k] if p["line"] else None
    return out


def _next(y, m, k):
    t = y * 12 + (m - 1) + k
    return t // 12, t % 12 + 1


def month_value(slug, month, fetch=True, max_ahead=2, upto=None):
    """Value of one CPCA line for one month (YYYY-MM) from the model pages.
    A page for month M lists the line's own row only when the line's M value is positive (observed on Gasgoo:
    a line with 0 / '-' in the queried month drops out of its own page). So try the pages M, M+1, M+2 (each shows
    3 months) and take the first page that has the line row - cached pages first, then live pages if fetch=True.
    upto: last page month allowed (YYYY-MM).
    -> (value or None, status, url). status: ok | dash ('-' shown) | not_positive (page M has no line row and no later
       page shows M) | missing (pages unavailable)"""
    y, m = int(month[:4]), int(month[5:])
    pages = []
    for k in range(0, max_ahead + 1):
        yy, mm = _next(y, m, k)
        if upto and f"{yy}-{mm:02d}" > upto:
            break
        pages.append((k, yy, mm))
    page0_seen, first_url = False, f"{BASE}/autosales/model/{slug}/{y}/{m}.html"
    for live in ((False, True) if fetch else (False,)):
        for k, yy, mm in pages:
            cached = _cached(f"m_{slug}_{yy}_{mm}.html") is not None
            if cached == live:      # pass 1: cached pages only; pass 2: pages not in cache
                continue
            p = model_page(slug, yy, mm)
            if k == 0 and p["status"] in ("ok", "line_no_value_3m"):
                page0_seen = True
            v = page_values(p)
            if p["line"] is not None and month in v:
                return v[month], ("ok" if v[month] is not None else "dash"), p["url"]
    return None, ("not_positive" if page0_seen else "missing"), first_url
