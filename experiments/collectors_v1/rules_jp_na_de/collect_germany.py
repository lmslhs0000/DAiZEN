"""Germany collector: KBA FZ10 (Neuzulassungen von Pkw nach Marken und Modellreihen) -> our (Brand, Model) series.

Usage
  python collect_germany.py --month 2026-07                 # download fz10_2026_07.xlsx if not cached, write output
  python collect_germany.py --month 2026-07 --offline       # cached files only
  python collect_germany.py --from 2023-01 --to 2026-07 --download-only   # fill the cache (30 s between files)

Rules
- File: https://www.kba.de/SharedDocs/Downloads/DE/Statistik/Fahrzeuge/FZ10/fz10_YYYY_MM.xlsx?__blob=publicationFile
  robots.txt: Crawl-delay 30 -> at least 30 s between requests to www.kba.de (state kept in cache/kba_state.json).
  HTTP 429 -> stop. Only 2019+ xlsx layout is parsed here (sheet 'FZ 10.1' / 'FZ10.1' = 4th sheet;
  col B brand carried down, col C model line, col D Insgesamt (month), col AK = 'mit Elektroantrieb (BEV)' month).
  '-' = 0. A leading 'r' (revised) is stripped.
- Mapping: data_audit_v1/germany_kba_mapping.csv (kba_brand + ' | '-separated KBA line aliases; values of all alias
  lines present in the month are summed).
- BEV split (split_rules.csv, rule 'kba_bev_column'): for KBA lines that hold both an ICE/PHEV series and a BEV series
  of ours (Audi Q8 / Q8 e-tron; BMW 4/i4, 5/i5, 7/i7, X1/iX1, X3/iX3), the BEV series = BEV column of the group lines,
  the other series = Insgesamt - BEV of the group lines.
Output CSV columns: Country, Month, Brand, Model, value, source_id, source_key, rule, status
"""
import argparse, csv, json, os, re, sys, time
import openpyxl
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
CACHE = os.path.join(HERE, 'cache', 'kba')
SAMPLES = os.path.join(ROOT, 'experiments', 'data_sources_v1', 'group_2', 'samples')
MAPPING = os.path.join(ROOT, 'experiments', 'data_audit_v1', 'germany_kba_mapping.csv')
URL = 'https://www.kba.de/SharedDocs/Downloads/DE/Statistik/Fahrzeuge/FZ10/fz10_{y}_{m:02d}.xlsx?__blob=publicationFile'
UA = 'Mozilla/5.0 (research; monthly statistics; contact via project owner)'
DELAY = 30.5

# (Brand, BEV member, other member) -> both map to the same KBA group lines
BEV_PAIRS = [
    ('Audi', 'Q8 e-tron', 'Q8', ['E-TRON', 'Q8']),
    ('BMW', 'i4', '4 Series', ['4ER']),
    ('BMW', 'i5', '5 Series', ['5ER']),
    ('BMW', 'i7', '7 Series', ['7ER']),
    ('BMW', 'iX1', 'X1', ['X1']),
    ('BMW', 'iX3', 'X3', ['IX3', 'X3']),
]


def cached_path(month):
    y, m = int(month[:4]), int(month[5:])
    name = f'fz10_{y}_{m:02d}.xlsx'
    for d in (CACHE, SAMPLES):
        p = os.path.join(d, name)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def download(month):
    os.makedirs(CACHE, exist_ok=True)
    st_path = os.path.join(CACHE, 'kba_state.json')
    st = json.load(open(st_path)) if os.path.exists(st_path) else {}
    if st.get('stopped'):
        raise RuntimeError('kba.de stopped earlier: ' + st['stopped'])
    wait = DELAY - (time.time() - st.get('last', 0))
    if wait > 0:
        time.sleep(wait)
    y, m = int(month[:4]), int(month[5:])
    url = URL.format(y=y, m=m)
    r = requests.get(url, headers={'User-Agent': UA}, timeout=90)
    st['last'] = time.time()
    if r.status_code == 429:
        st['stopped'] = 'HTTP 429 ' + time.strftime('%Y-%m-%d %H:%M')
    json.dump(st, open(st_path, 'w'))
    with open(os.path.join(CACHE, 'fetch_log.csv'), 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([time.strftime('%Y-%m-%d %H:%M:%S'), url, r.status_code, len(r.content)])
    if r.status_code != 200 or not r.content.startswith(b'PK'):
        print('FAIL', r.status_code, url, file=sys.stderr)
        return None
    p = os.path.join(CACHE, f'fz10_{y}_{m:02d}.xlsx')
    open(p, 'wb').write(r.content)
    return p


def num(v):
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s in ('-', '–', '', '.'):
        return 0.0
    if s == '/':  # KBA suppresses small cells ('/' = value withheld)
        return None
    s = re.sub(r'^r\s*', '', s).replace('.', '').replace(' ', '')
    try:
        return float(s)
    except ValueError:
        return None


def parse_fz10(path):
    """-> DataFrame kba_brand, kba_model, total, bev"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sn = [s for s in wb.sheetnames if re.sub(r'\s', '', s).upper().startswith('FZ10.1')]
    ws = wb[sn[0] if sn else wb.sheetnames[3]]
    rows = list(ws.iter_rows(values_only=True))
    hdr = next(i for i, r in enumerate(rows) if r[1] == 'Marke')
    top = next(i for i in range(hdr, max(hdr - 3, 0), -1) if any(str(c or '').startswith('Insgesamt') for c in rows[i]))
    head = [str(c or '') for c in rows[top]]
    ci = next(i for i, c in enumerate(head) if c.startswith('Insgesamt'))
    cb = next(i for i, c in enumerate(head) if 'Elektroantrieb (BEV)' in c)
    hdr = max(hdr, top) - 1  # data rows start two rows below the 'Insgesamt' header row
    out, brand = [], None
    for r in rows[hdr + 2:]:
        if r[1] not in (None, ''):
            brand = str(r[1]).strip()
        if brand is None or 'ZUSAMMEN' in brand or r[2] in (None, ''):
            continue
        if brand.upper().startswith('INSGESAMT') or brand.upper().startswith('SONSTIGE'):
            continue
        out.append(dict(kba_brand=brand, kba_model=re.sub(r'\s+', ' ', str(r[2])).strip(),
                        total=num(r[ci]), bev=num(r[cb])))
    return pd.DataFrame(out)


def collect(month, offline=False):
    p = cached_path(month)
    if p is None and not offline:
        p = download(month)
    if p is None:
        raise SystemExit(f'no FZ10 file for {month}')
    k = parse_fz10(p)
    mp = pd.read_csv(MAPPING)
    lines = {(b, m): (t, e) for b, m, t, e in k[['kba_brand', 'kba_model', 'total', 'bev']].itertuples(index=False)}
    pair_member = {}
    for brand, ev, ice, kl in BEV_PAIRS:
        pair_member[(brand, ev)] = ('bev', kl)
        pair_member[(brand, ice)] = ('rest', kl)
    out = []
    for r in mp.itertuples(index=False):
        aliases = [a.strip() for a in str(r.kba_model_series).split(' | ')]
        rule, kl = 'sum_alias_lines', aliases
        if (r.Brand, r.Model) in pair_member:
            kind, kl = pair_member[(r.Brand, r.Model)]
            rule = 'kba_bev_column:' + ('BEV' if kind == 'bev' else 'total_minus_BEV')
        found = [lines[(r.kba_brand, a)] for a in kl if (r.kba_brand, a) in lines]
        if not found:
            val, status = 0.0, 'LINE_ABSENT_0'  # KBA lists no line -> no registrations that month
        else:
            tot = sum(f[0] for f in found if f[0] is not None)
            bev = sum(f[1] for f in found if f[1] is not None)
            if rule.endswith('BEV') and 'minus' not in rule:
                val = bev
            elif 'minus' in rule:
                val = tot - bev
            else:
                val = tot
            status = 'OK'
        out.append(dict(Country='Germany', Month=month, Brand=r.Brand, Model=r.Model, value=val,
                        source_id='DE_KBA_FZ10', source_key=f'{r.kba_brand}/' + '+'.join(kl), rule=rule,
                        status=status, file=os.path.basename(p)))
    # alert: KBA lines not used by any mapping row
    used = {(r.kba_brand, a.strip()) for r in mp.itertuples(index=False) for a in str(r.kba_model_series).split(' | ')}
    for b, e, ice, kl in BEV_PAIRS:
        used |= {(mp[(mp.Brand == b)].kba_brand.iloc[0], x) for x in kl}
    mask = [bool((b, x) not in used and x != 'SONSTIGE' and t is not None and t > 0) for b, x, t in
            k[['kba_brand', 'kba_model', 'total']].itertuples(index=False)]
    unused = k[mask]
    return pd.DataFrame(out), unused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--month')
    ap.add_argument('--from', dest='frm')
    ap.add_argument('--to')
    ap.add_argument('--offline', action='store_true')
    ap.add_argument('--download-only', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    months = [a.month] if a.month else [str(p) for p in pd.period_range(a.frm, a.to, freq='M')]
    if a.download_only:
        for m in months:
            if cached_path(m) is None:
                print(m, download(m), flush=True)
        return
    res = []
    for m in months:
        d, unused = collect(m, a.offline)
        res.append(d)
        if len(unused):
            print(f'{m}: {len(unused)} KBA lines with sales not in mapping (new models?):',
                  '; '.join(f'{b}/{x}={int(t)}' for b, x, t in list(unused[['kba_brand', 'kba_model', 'total']].itertuples(index=False))[:15]))
    d = pd.concat(res)
    out = a.out or os.path.join(HERE, 'out', f'germany_{months[0]}_{months[-1]}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False, encoding='utf-8-sig')
    print('wrote', out, len(d))


if __name__ == '__main__':
    main()
