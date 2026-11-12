"""group_0 (China) source probe: fetch one URL per call list, per-host >=2.5 s gap, stop host on 429.
usage: python probe.py name=url [name=url ...]  -> samples/<name>, log line to probe_log.txt"""
import sys, time, urllib.request, urllib.error, urllib.parse, ssl
from pathlib import Path
H = Path(__file__).resolve().parent; S = H / "samples"; S.mkdir(exist_ok=True)
LAST = {}; BLOCKED = set()
ctx = ssl.create_default_context()
def fetch(name, url):
    host = urllib.parse.urlparse(url).netloc
    if host in BLOCKED: return "skip-429"
    w = 2.5 - (time.time() - LAST.get(host, 0))
    if w > 0: time.sleep(w)
    LAST[host] = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept-Language": "en,zh;q=0.8"})
    try:
        with urllib.request.urlopen(req, timeout=40, context=ctx) as r:
            b = r.read(); st = r.status; ct = r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        st = e.code; ct = ""; b = e.read() if hasattr(e, "read") else b""
        if st == 429: BLOCKED.add(host)
    except Exception as e:
        st = type(e).__name__; ct = ""; b = b""
    (S / name).write_bytes(b)
    msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{st}\t{len(b)}\t{ct}\t{name}\t{url}"
    with open(H / "probe_log.txt", "a", encoding="utf-8") as f: f.write(msg + "\n")
    print(msg, flush=True); return st
for a in sys.argv[1:]:
    n, u = a.split("=", 1); fetch(n, u)
