"""Fetch every URL in ca_urls.csv once into cache/ (polite, 2.5 s gap). Re-running skips cached pages."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gcbc_fetch

if __name__ == "__main__":
    u = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ca_urls.csv"), encoding="utf-8-sig")
    urls = list(dict.fromkeys(x for x in u.url.fillna("") if x))
    for i, url in enumerate(urls):
        try:
            s, h = gcbc_fetch.get(url)
        except RuntimeError as e:
            print("STOP", e, flush=True)
            break
        print(i, s, len(h), url, flush=True)
