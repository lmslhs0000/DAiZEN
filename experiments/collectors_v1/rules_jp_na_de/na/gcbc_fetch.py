"""Cached polite fetch for GoodCarBadCar (and other hosts), reusing group_1/scripts/polite.py.
State/log files are kept inside this folder. Pages are cached in cache/<slug>.html."""
import csv, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("POLITE_STATE", os.path.join(HERE, "polite_state.json"))
os.environ.setdefault("POLITE_LOG", os.path.join(HERE, "fetch_log.csv"))
G1 = os.path.join(HERE, "..", "..", "..", "data_sources_v1", "group_1", "scripts")
sys.path.insert(0, os.path.abspath(G1))
import polite  # noqa: E402

CACHE = os.path.join(HERE, "cache")
GROUP1_SAMPLES = os.path.abspath(os.path.join(G1, "..", "samples"))


def cache_path(url):
    slug = re.sub(r"^https?://[^/]+/|/$", "", url)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", slug) or "index"
    return os.path.join(CACHE, slug + ".html")


def log(url, rv, status, size, path):
    new = not os.path.exists(polite.LOG)
    with open(polite.LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "url", "robots", "status", "bytes", "file"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), url, rv, status, size, os.path.basename(path)])


def get(url, offline=False, refresh=False):
    """Return (status, html). status: 'cache' | HTTP code str | 'ROBOTS_DISALLOWED' | 'OFFLINE_MISS' | 'ERR ...'"""
    p = cache_path(url)
    if os.path.exists(p) and not refresh:
        return "cache", open(p, encoding="utf-8").read()
    if offline:
        return "OFFLINE_MISS", ""
    st = polite._state()
    rv = polite.robots(url, st)
    if rv == "no":
        log(url, rv, "ROBOTS_DISALLOWED", 0, p)
        return "ROBOTS_DISALLOWED", ""
    try:
        r = polite._raw_get(url, st)
    except Exception as e:  # noqa
        log(url, rv, f"ERR {e}"[:120], 0, p)
        return f"ERR {e}"[:120], ""
    status = str(r.status_code)
    html = ""
    if r.status_code == 200:
        html = r.text
        os.makedirs(CACHE, exist_ok=True)
        open(p, "w", encoding="utf-8").write(html)
    log(url, rv, status, len(r.content), p)
    if r.status_code == 429:
        raise RuntimeError("HTTP 429 from " + url + " -> host stopped")
    return status, html


if __name__ == "__main__":
    for u in sys.argv[1:]:
        s, h = get(u)
        print(s, len(h), u)
