"""Netherlands (RDW Open Data, Gekentekende voertuigen m9d7-ebf2) -> new passenger cars by make/model/month.

Usage:
    python collect_nl.py --origin 2026-07 [--months 4] [--out out/nl_model_month.csv]

- One Socrata (SoQL) aggregate request per month, >= 2 s apart (robots.txt Crawl-delay 1).
  Raw JSON responses are cached in cache/nl/.
- New passenger car filter:
    voertuigsoort = 'Personenauto'
    AND datum_eerste_toelating_dt (first admission anywhere) in the month
    AND datum_eerste_toelating_dt = datum_eerste_tenaamstelling_in_nederland_dt
  (= first admitted in NL, i.e. not a used import). 2025 sum of this filter = 388,567 vs
  ACEA 2025 Netherlands 388,024 (+0.14 %).
- Caveat: the dataset is the *current* register; cars exported or scrapped after
  registration may disappear over time, so old months can drift down slightly.
- Output columns: country, month, make, model, units  (make=merk, model=handelsbenaming).
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache', 'nl')
BASE = 'https://opendata.rdw.nl/resource/m9d7-ebf2.json'
UA = 'daizen-research/1.0 (monthly statistics collector)'


def month_range(origin, n):
    y, m = map(int, origin.split('-'))
    out = []
    for _ in range(n):
        out.append((y, m))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return sorted(out)


def next_month(y, m):
    return (y + 1, 1) if m == 12 else (y, m + 1)


def fetch_month(y, m, refresh=False, last=[0.0]):
    path = os.path.join(CACHE, f'rdw_new_m1_{y}{m:02d}.json')
    if os.path.exists(path) and not refresh:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    ny, nm = next_month(y, m)
    where = ("voertuigsoort='Personenauto' AND "
             f"datum_eerste_toelating_dt >= '{y}-{m:02d}-01T00:00:00' AND "
             f"datum_eerste_toelating_dt < '{ny}-{nm:02d}-01T00:00:00' AND "
             "datum_eerste_toelating_dt = datum_eerste_tenaamstelling_in_nederland_dt")
    q = {'$select': 'merk, handelsbenaming, count(*) as n', '$where': where,
         '$group': 'merk, handelsbenaming', '$order': 'n DESC', '$limit': '50000'}
    url = BASE + '?' + urllib.parse.urlencode(q)
    wait = 2.0 - (time.time() - last[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        if e.code == 429:
            sys.exit('HTTP 429 from opendata.rdw.nl -> stop (site etiquette)')
        raise
    finally:
        last[0] = time.time()
    if len(data) >= 50000:
        sys.exit('result hit $limit; paginate')
    os.makedirs(CACHE, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--origin', required=True)
    ap.add_argument('--months', type=int, default=1)
    ap.add_argument('--refresh', action='store_true', help='ignore cache (register changes over time)')
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'nl_model_month.csv'))
    a = ap.parse_args()
    rows = []
    for y, m in month_range(a.origin, a.months):
        data = fetch_month(y, m, a.refresh)
        tot = 0
        for d in data:
            n = int(d['n'])
            tot += n
            rows.append(('Netherlands', f'{y}-{m:02d}', d.get('merk', '').strip(),
                         d.get('handelsbenaming', '').strip(), n))
        print(f'{y}-{m:02d}: {len(data)} make/model keys, {tot} cars', file=sys.stderr)
    rows.sort(key=lambda r: (r[1], -r[4], r[2], r[3]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['country', 'month', 'make', 'model', 'units'])
        w.writerows(rows)
    print('wrote', a.out, len(rows), 'rows', file=sys.stderr)


if __name__ == '__main__':
    main()
