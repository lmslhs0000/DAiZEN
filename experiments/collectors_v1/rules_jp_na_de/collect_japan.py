"""Japan collector: 全軽自協 (Zenkeijikyo) kei by-name xls + JADA registered-passenger top-50 xlsx -> our (Brand, Model) values.

Usage
  python collect_japan.py --month 2026-07                      # find/download files, write out/japan_2026-07.csv
  python collect_japan.py --month 2026-07 --offline            # cached files only
  python collect_japan.py --from 2024-01 --to 2026-07 --offline --out work/jp_hist.csv   # history (validation)
  options: --zenkei FILE.xls  --jada FILE.xlsx [FILE.xlsx ...]  --raw D:/daizen/raw/data.csv

Rules (mapping.csv, Country == Japan; split_rules.csv)
  direct              one official line = our series
  sum                 several official kei lines (segment|maker|name aliases) are added (e.g. Hijet = cargo + e-cargo + truck)
  direct+fill_window  JADA 1:1 line; if the model is outside the monthly top 50, the month is filled from the window
                      lists in the same workbooks (Jan-Jun, Apr-Sep, Jan-Dec, fiscal Apr-Mar):
                        residual = window total - months listed in the window
                        one unknown month in a window -> exact (fill_exact; solved iteratively across windows)
                        several unknown months -> residual allocated with our same-month values of the previous year
                        (equal if none), capped at that month's rank-50 value (fill_alloc = allocation, not observed)
                      not in any list -> value empty, status NOT_IN_TOP50
  split:<group>       JADA bundles one line for several of our rows (Yaris+Yaris Cross, Corolla+Corolla Cross, Crown x4,
                      Land Cruiser+FJ, Note+AURA, Impreza+Crosstrek). Value = JADA total x share of the member in our raw
                      data over the trailing N months before the target month (N per group in split_rules.csv).
                      status ALLOCATED (an assumption, not an observation).
Output columns: Country, Month, Brand, Model, value, rule, status, source_id, source_key, official_total, note
Only months <= 2026-07 may be written into our data (project rule); later months are format checks only.
"""
import argparse, glob, json, os, re, sys, time, unicodedata
import openpyxl, xlrd
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
CACHE = os.path.join(HERE, 'cache', 'japan')
AUDIT_CACHE = os.path.join(ROOT, 'experiments', 'data_audit_v1', 'japan', 'cache')
G3 = os.path.join(ROOT, 'experiments', 'data_sources_v1', 'group_3', 'samples')
RAW = os.path.join(ROOT, 'raw', 'data.csv')
MAPPING = os.path.join(HERE, 'mapping.csv')
UA = 'Mozilla/5.0 (research; monthly statistics)'
GAP = 3.0
SPLIT_TRAIL = {'TOY_YARIS': 1, 'TOY_COROLLA': 3, 'TOY_CROWN': 1, 'TOY_LANDCRUISER': 1, 'NIS_NOTE': 3, 'SUB_IMPREZA': 1}
JADA_OLD = {  # yearly workbooks (file ids change when JADA re-uploads; current year is read from /pages/340/)
    2024: 'https://www.jada.or.jp/files/libs/6123/202509241132264821.xlsx',
    2025: 'https://www.jada.or.jp/files/libs/6987/202604031138204699.xlsx',
}
MAKER = {'ミツビシ': '三菱', 'ニッサン': '日産'}


def N(s):
    return unicodedata.normalize('NFKC', str(s)).strip()


def KEY(s):
    return re.sub(r'[-ー－\s]', '', N(s).upper())


# ---------------------------------------------------------------- polite download
_last = {}


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    host = re.sub(r'^https?://([^/]+).*', r'\1', url)
    w = GAP - (time.time() - _last.get(host, 0))
    if w > 0:
        time.sleep(w)
    r = requests.get(url, headers={'User-Agent': UA}, timeout=60)
    _last[host] = time.time()
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, 'fetch_log.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(dict(url=url, status=r.status_code, bytes=len(r.content), time=time.strftime('%F %T'))) + '\n')
    if r.status_code == 429:
        sys.exit('HTTP 429 from ' + host + ' -> stop')
    if r.status_code != 200:
        print('FAIL', r.status_code, url, file=sys.stderr)
        return None
    open(path, 'wb').write(r.content)
    return path


def find_zenkei(month, offline):
    """確報 xls for the month: cache first, then tushokaku index page -> month page -> xls (速報 page as fallback)."""
    pats = [os.path.join(CACHE, f'zenkei_{month}__*.xls*'), os.path.join(AUDIT_CACHE, 'zenkei', f'{month}__*.xls*')]
    for p in pats:
        hit = sorted(glob.glob(p))
        if hit:
            return hit[-1], 'cache'
    yymm = month[2:4] + month[5:7]
    for f in glob.glob(os.path.join(G3, 'zenkei_*.xls')):
        if f.endswith(yymm + '.xls'):
            return f, 'sample'
    if offline:
        return None, ''
    idx = fetch('https://www.zenkeijikyo.or.jp/statistics/tushokaku',
                os.path.join(CACHE, f'tushokaku_index_{time.strftime("%Y%m%d")}.html'))
    h = open(idx, encoding='utf-8', errors='replace').read()
    y, m = int(month[:4]), int(month[5:])
    pid = None
    for sec in re.finditer(r'<h2>(\d{4})年</h2>(.*?)</section>', h, re.S):
        if int(sec.group(1)) == y:
            for mm in re.finditer(r'statistics/tushokaku-(\d+)">(\d+)</a>', sec.group(2)):
                if int(mm.group(2)) == m:
                    pid = mm.group(1)
    kind = '確報'
    if pid:
        page = fetch(f'https://www.zenkeijikyo.or.jp/statistics/tushokaku-{pid}', os.path.join(CACHE, f'tushokaku-{pid}.html'))
    else:  # 確報 not out yet -> 速報 page (only holds the latest month)
        page = fetch('https://www.zenkeijikyo.or.jp/statistics/tushosoku', os.path.join(CACHE, f'tushosoku_{time.strftime("%Y%m%d")}.html'))
        kind = '速報'
    links = re.findall(r'href="([^"]+\.xlsx?)"', open(page, encoding='utf-8', errors='replace').read())
    links = [u for u in links if yymm in u.rsplit('/', 1)[1]] or links
    if not links:
        return None, ''
    u = links[0]
    return fetch(u, os.path.join(CACHE, f'zenkei_{month}__' + u.rsplit('/', 1)[1])), kind


def find_jada(month, offline):
    """Workbooks for the target year and the previous year (window lists cross years via fiscal year)."""
    y = int(month[:4])
    out = []
    for yy in (y - 1, y):
        cands = sorted(glob.glob(os.path.join(CACHE, f'jada_rank_{yy}*.xlsx')))
        if yy == 2026:
            cands += [os.path.join(G3, 'jada_rank_2026_ytd_20261005.xlsx')]
        cands += [os.path.join(AUDIT_CACHE, 'jada', f'jada_rank_{yy}.xlsx')]
        cands = [c for c in cands if os.path.exists(c)]
        if cands and (offline or yy < y):
            out.append(cands[0])
            continue
        if offline:
            continue
        if yy in JADA_OLD:
            p = fetch(JADA_OLD[yy], os.path.join(CACHE, f'jada_rank_{yy}.xlsx'))
        else:
            pg = fetch('https://www.jada.or.jp/pages/340/', os.path.join(CACHE, f'jada_pages_340_{time.strftime("%Y%m%d")}.html'))
            links = re.findall(r'href="(https://www\.jada\.or\.jp/files/libs/\d+/\d+\.xlsx)"', open(pg, encoding='utf-8').read())
            p = fetch(links[0], os.path.join(CACHE, f'jada_rank_{yy}_{time.strftime("%Y%m%d")}.xlsx')) if links else None
        if p:
            out.append(p)
    return out


# ---------------------------------------------------------------- parsers
def read_zenkei(fn):
    """second block of the xls (all models): segment (carried), maker (carried), name, 本月."""
    sh = xlrd.open_workbook(fn).sheet_by_index(0)
    hdr = [i for i in range(sh.nrows) if N(sh.cell_value(i, 0)) == '車種' and '通称' in N(sh.cell_value(i, 2))]
    h = hdr[-1]
    heads = [N(c) for c in sh.row_values(h)]
    ci = heads.index('本月')
    rows, sec, maker = [], None, None
    for i in range(h + 1, sh.nrows):
        r = sh.row_values(i)
        if all(c == '' for c in r):
            continue
        if any(str(c).strip().startswith(('※', '＊')) for c in r):
            break
        if N(r[0]) != '':
            sec = re.sub(r'[\s\d]', '', N(r[0]))
        if N(r[1]) != '':
            maker = N(r[1])
        name = N(r[2]) if not isinstance(r[2], float) else str(int(r[2]))
        if not name:
            continue
        v = r[ci]
        rows.append(dict(segment=sec, maker=MAKER.get(maker, maker), key=KEY(name), name=name,
                         units=float(v) if isinstance(v, float) else (0.0 if v == '' else None)))
    return pd.DataFrame(rows)


def months_of_window(label):
    s = N(label)
    m = re.fullmatch(r'(\d{4})年(\d{1,2})月', s)
    if m:
        return 'month', [f'{int(m.group(1))}-{int(m.group(2)):02d}']
    m = re.fullmatch(r'(\d{4})年度', s)
    if m:
        y = int(m.group(1))
        return 'window', [str(p) for p in pd.period_range(f'{y}-04', f'{y + 1}-03', freq='M')]
    m = re.fullmatch(r'(\d{4})年(\d{1,2})月?[~～〜](\d{1,2})月', s)
    if m:
        y = int(m.group(1))
        return 'window', [str(p) for p in pd.period_range(f'{y}-{int(m.group(2)):02d}', f'{y}-{int(m.group(3)):02d}', freq='M')]
    return None, []


def read_jada(files):
    """-> months: {period: DataFrame(rank, maker, key, units)}, windows: [(months, DataFrame)]"""
    months, windows = {}, []
    for fn in files:
        wb = openpyxl.load_workbook(fn, data_only=True, read_only=True)
        for sn in wb.sheetnames:
            kind, ms = months_of_window(sn)
            if not kind:
                continue
            rows = []
            for r in wb[sn].iter_rows(values_only=True):
                if len(r) > 4 and isinstance(r[1], int) and r[2] and isinstance(r[4], (int, float)):
                    mk = N(r[3])
                    rows.append(dict(rank=r[1], maker=MAKER.get(mk, mk), key=KEY(r[2]), name=N(r[2]), units=float(r[4])))
            df = pd.DataFrame(rows)
            if kind == 'month':
                months.setdefault(ms[0], df)
            elif not any(w[0] == ms for w in windows):
                windows.append((ms, df))
    return months, windows


# ---------------------------------------------------------------- rules
def parse_key(k):
    src, seg, maker, names = k.split('|')
    return src, seg, maker, [KEY(x) for x in names.split(';')]


def jada_value(months, period, maker, keys):
    df = months.get(period)
    if df is None or df.empty:
        return None
    x = df[(df.maker == maker) & df.key.isin(keys)]
    return float(x.units.sum()) if len(x) else None


def fill_window(target, maker, keys, months, windows, raw_series):
    """Fill a month missing from the monthly top 50 from window totals. Returns (value, status, note)."""
    cons = []
    for ms, df in windows:
        if target not in ms:
            continue
        x = df[(df.maker == maker) & df.key.isin(keys)]
        if not len(x):
            continue
        if any(m not in months for m in ms):  # window covers a month whose monthly list we do not have
            continue
        known, unknown = 0.0, []
        for m in ms:
            v = jada_value(months, m, maker, keys)
            if v is None:
                unknown.append(m)
            else:
                known += v
        cons.append([set(unknown), float(x.units.sum()) - known, '+'.join([ms[0], ms[-1]])])
    if not cons:
        return None, 'NOT_IN_TOP50', 'not in monthly or window top-50 lists'
    solved = {}
    changed = True
    while changed:  # propagate windows that have one unknown month left
        changed = False
        for c in cons:
            u = c[0] - set(solved)
            if len(u) == 1:
                m = next(iter(u))
                solved[m] = c[1] - sum(solved[s] for s in c[0] if s in solved)
                changed = True
    if target in solved:
        return solved[target], 'FILLED_EXACT', 'window total minus listed months'
    c = min(cons, key=lambda c: len(c[0] - set(solved)))
    u = sorted(c[0] - set(solved))
    resid = c[1] - sum(solved[s] for s in c[0] if s in solved)
    w = {m: raw_series.get(f'{int(m[:4]) - 1}{m[4:]}', 0) or 0 for m in u}
    if sum(w.values()) <= 0:
        w = {m: 1.0 for m in u}
    caps = {m: (months[m].units.min() if m in months and len(months[m]) >= 50 else float('inf')) for m in u}
    alloc = {m: resid * w[m] / sum(w.values()) for m in u}
    # cap at the rank-50 value of the month and pass the excess to uncapped months
    for _ in range(5):
        over = {m: alloc[m] - caps[m] for m in u if alloc[m] > caps[m]}
        if not over:
            break
        free = [m for m in u if m not in over]
        for m in over:
            alloc[m] = caps[m]
        if not free:
            break
        tw = sum(w[m] for m in free) or len(free)
        for m in free:
            alloc[m] += sum(over.values()) * (w[m] / tw if tw else 1 / len(free))
    return alloc[target], 'FILLED_ALLOC', f'window {c[2]} residual {resid:.0f} over {len(u)} months (weights: our same month last year, cap rank-50)'


def collect(month, zenkei_file, jada_files, raw):
    mp = pd.read_csv(MAPPING)
    mp = mp[mp.Country == 'Japan'].fillna('')
    kei = read_zenkei(zenkei_file) if zenkei_file else pd.DataFrame(columns=['segment', 'maker', 'key', 'units'])
    months, windows = read_jada(jada_files) if jada_files else ({}, [])
    rawj = raw[(raw.Country == 'Japan') & (raw.Month < month)]
    piv = rawj.pivot_table(index='Month', columns=['Brand', 'Model'], values='Sales', aggfunc='sum')
    out = []
    for (b, m), g in mp.groupby(['Brand', 'Model'], sort=False):
        rule = g.rule.iloc[0]
        rec = dict(Country='Japan', Month=month, Brand=b, Model=m, value=None, rule=rule, status='', source_id=g.source_id.iloc[0],
                   source_key=' + '.join(g.source_key), official_total=None, note='')
        if rule == 'none':
            rec['status'] = 'NO_SOURCE'
            rec['note'] = g.note.iloc[0]
            out.append(rec)
            continue
        vals, miss = [], []
        for k in g.source_key:
            src, seg, maker, keys = parse_key(k)
            if src == 'KEI':
                if kei.empty:
                    miss.append(k)
                    continue
                x = kei[(kei.segment == seg) & (kei.maker == maker) & kei.key.isin(keys)]
                vals.append(float(x.units.sum()))  # a line absent from the full list = 0 sold that month
            else:
                v = jada_value(months, month, maker, keys)
                if v is None and month in months:
                    raw_s = piv[(b, m)] if (b, m) in piv.columns else pd.Series(dtype=float)
                    v, st, note = fill_window(month, maker, keys, months, windows, raw_s)
                    rec['status'], rec['note'] = st, note
                if v is None:
                    miss.append(k)
                else:
                    vals.append(v)
        if miss:
            rec['status'] = rec['status'] or 'SOURCE_MISSING'
            out.append(rec)
            continue
        total = sum(vals)
        rec['official_total'] = total
        if rule.startswith('split:'):
            gid = rule.split(':')[1]
            members = mp[mp.rule == rule][['Brand', 'Model']].drop_duplicates()
            mem = [tuple(x) for x in members.itertuples(index=False)]
            n = SPLIT_TRAIL[gid]
            cols = [c for c in mem if c in piv.columns]
            hist = piv[cols].tail(n).fillna(0).sum() if cols else pd.Series(dtype=float)
            if hist.sum() <= 0:
                rec['status'] = 'NO_SHARE_HISTORY'
                out.append(rec)
                continue
            share = float(hist.get((b, m), 0.0)) / float(hist.sum())
            rec['value'] = total * share
            rec['status'] = 'ALLOCATED'
            rec['note'] = f'share {share:.4f} = our trailing {n}-month sales before {month}; JADA bundle total {total:.0f}'
        else:
            rec['value'] = total
            rec['status'] = rec['status'] or 'OK'
        out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--month')
    ap.add_argument('--from', dest='frm')
    ap.add_argument('--to')
    ap.add_argument('--zenkei')
    ap.add_argument('--jada', nargs='*')
    ap.add_argument('--raw', default=RAW)
    ap.add_argument('--offline', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    raw = pd.read_csv(a.raw)
    months = [a.month] if a.month else [str(p) for p in pd.period_range(a.frm, a.to, freq='M')]
    res = []
    for mo in months:
        zf, kind = (a.zenkei, 'given') if a.zenkei else find_zenkei(mo, a.offline)
        jf = a.jada if a.jada else find_jada(mo, a.offline)
        d = collect(mo, zf, jf, raw)
        d['files'] = f'zenkei={os.path.basename(zf) if zf else None}({kind}); jada=' + ','.join(os.path.basename(x) for x in jf)
        res.append(d)
        print(mo, d.status.value_counts().to_dict(), flush=True)
    d = pd.concat(res)
    out = a.out or os.path.join(HERE, 'out', f'japan_{months[0]}_{months[-1]}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False, encoding='utf-8-sig')
    print('wrote', out, len(d))


if __name__ == '__main__':
    main()
