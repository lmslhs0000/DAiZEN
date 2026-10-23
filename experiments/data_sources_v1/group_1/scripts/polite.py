"""Polite fetcher: robots.txt check, >=2.5 s between requests to the same host (persisted across runs),
stop a host after HTTP 429. Log every request to fetch_log.csv. Usage:
  python polite.py URL OUTFILE      -> fetch one URL into samples/OUTFILE
  python polite.py --robots URL     -> print robots verdict only
"""
import csv, json, os, sys, time, urllib.parse, urllib.robotparser
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.dirname(HERE)
SAMPLES = os.path.join(G, "samples")
STATE = os.environ.get("POLITE_STATE", os.path.join(HERE, "host_state.json"))
LOG = os.environ.get("POLITE_LOG", os.path.join(HERE, "fetch_log.csv"))
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
GAP = 2.5


def _state():
    return json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {}


def _save(st):
    json.dump(st, open(STATE, "w", encoding="utf-8"), indent=1)


def _wait(host, st):
    last = st.get(host, {}).get("last", 0)
    d = time.time() - last
    if d < GAP:
        time.sleep(GAP - d)


def _raw_get(url, st):
    host = urllib.parse.urlparse(url).netloc
    if st.get(host, {}).get("stopped"):
        raise RuntimeError(f"host {host} stopped earlier ({st[host]['stopped']})")
    _wait(host, st)
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.8"}, timeout=40)
    finally:
        st.setdefault(host, {})["last"] = time.time()
        _save(st)
    if r.status_code == 429:
        st[host]["stopped"] = "HTTP 429"
        _save(st)
    return r


def robots(url, st=None):
    st = st if st is not None else _state()
    p = urllib.parse.urlparse(url)
    host = p.netloc
    cache = st.get(host, {}).get("robots")
    if cache is None:
        try:
            r = _raw_get(f"{p.scheme}://{host}/robots.txt", st)
            cache = r.text if r.status_code == 200 else f"__status_{r.status_code}"
        except Exception as e:  # noqa
            cache = f"__error_{e}"
        st.setdefault(host, {})["robots"] = cache
        _save(st)
    if cache.startswith("__status_4"):
        return "yes(no robots.txt)"
    if cache.startswith("__"):
        return f"unknown({cache[2:40]})"
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(cache.splitlines())
    return "yes" if rp.can_fetch("*", url) else "no"


def fetch(url, out):
    st = _state()
    rv = robots(url, st)
    status, size = "", 0
    if rv == "no":
        status = "ROBOTS_DISALLOWED"
    else:
        try:
            r = _raw_get(url, st)
            status = str(r.status_code)
            if r.status_code == 200 and out:
                os.makedirs(SAMPLES, exist_ok=True)
                open(os.path.join(SAMPLES, out), "wb").write(r.content)
                size = len(r.content)
        except Exception as e:  # noqa
            status = f"ERR {e}"[:120]
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "url", "robots", "status", "bytes", "file"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), url, rv, status, size, out])
    print(rv, status, size, out)
    return status


if __name__ == "__main__":
    if sys.argv[1] == "--robots":
        for u in sys.argv[2:]:
            print(u, robots(u))
    else:
        args = sys.argv[1:]
        for i in range(0, len(args), 2):
            fetch(args[i], args[i + 1])
