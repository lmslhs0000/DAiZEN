"""South Korea collector -> standard output (SPEC2.md).

Usage
  python collect_korea.py --month 2026-07 --offline          # cached files only (korea/cache/...)
  python collect_korea.py --month 2026-07                    # fetch what is missing (polite, see below)
  options: --hmc-file F.xlsx  --kgm-file F.xlsx  --raw D:/daizen/raw/data.csv  --no-write
Prints the path of out/std_korea_<YYYY-MM>.csv as the last line.

Sources (mapping.csv, lowest priority number with a value wins; priority 9 = check only)
  KR_HMC_IR           Hyundai IR 'Unit Sales by Model' xlsx, Domestic block (Hyundai + Genesis cars/RVs). ~30-45 days lag.
                      Until the file covering the month is out, Danawa is used (confidence medium, note says so).
  KR_KGM_IR           KG Mobility IR xlsx (sheet YYYY, Domestic CBU rows; ICE + HEV/EV rows added). The download link
                      is only given by a site API that needs a session token, so the file is put in cache/kgm_ir/
                      by hand (or --kgm-file); without it Danawa is used.
  KR_HMG_PRESS        Hyundai/Kia newsroom release (day 1). Primary only for Kia Bongo 3 (봉고Ⅲ); checks otherwise.
                      URL slug {hyundai-motor-company|kia}-{month}-{year}-sales works from 2026-05 (2026-04 -> HTTP 410;
                      older releases have CONT... ids). A release saved by hand as cache/press/<slug>.html is used too.
  KR_DANAWA_DOMESTIC  auto.danawa.com sales record, domestic makers (Kia, Hyundai CVs, Renault Korea, GM Korea + fallback)
  KR_DANAWA_IMPORT    auto.danawa.com sales record, imports (KAIDA registrations)
  KR_KAIDA_PRESS      KAIDA monthly release brand totals: checks only
Danawa lists only lines with sales in the month, so a mapped line that is absent counts as 0.
Lines with sales that are not in mapping.csv: a unique name-variant match inside the same Danawa brand
(e.g. '디 올 뉴 아반떼' -> '아반떼') is used with a note; everything else goes to out/unmapped_korea_<month>.csv.

Network rules: robots.txt is read and obeyed; same host >= 3 s, Danawa >= 10.5 s (Crawl-delay 10); HTTP 429 -> that
site is stopped for the run (cache/BLOCKED_<host>.txt). curl is used because Python's TLS handshake fails on Danawa.
Months after 2026-07 are format checks only (project rule: they do not go into our data).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.robotparser

import pandas as pd

import kr_parse as K

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
CACHE = os.path.join(HERE, 'cache')
OUT = os.path.join(HERE, 'out')
RAW = os.path.join(ROOT, 'raw', 'data.csv')
LAST_ALLOWED = '2026-07'
UA = 'Mozilla/5.0 (research; monthly statistics)'
MONTH_NAMES = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
               'november', 'december']
HMC_BASE = ('https://www.hyundai.com/content/dam/hyundai/ww/en/images/company/investor-relations/ir-resources/'
            'sales-results-report/')
DANAWA = 'https://auto.danawa.com/auto/?Work=record&Tab=Model&Nation={nat}&Month={m}-00'
STD_COLS = ['Country', 'Brand', 'Model', 'Month', 'Sales', 'value_type', 'estimated_share', 'source_id', 'confidence', 'note']


# ================================================================== polite fetch (curl)
class Blocked(Exception):
    pass


_last, _robots = {}, {}


def _host(url):
    return re.sub(r'^https?://([^/]+).*', r'\1', url)


def _curl(url, path, data=None):
    cmd = ['curl', '-s', '-L', '-A', UA, '--max-time', '90', '-o', path, '-w', '%{http_code}', url]
    if data:
        cmd[1:1] = ['--data', data]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(r.stdout.strip()[-3:])
    except ValueError:
        return 0


def _robots_ok(url):
    host = _host(url)
    if host not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        tmp = os.path.join(CACHE, f'robots_{host}.txt')
        code = _curl(f'https://{host}/robots.txt', tmp)
        _last[host] = time.time()
        txt = open(tmp, encoding='utf-8', errors='ignore').read() if os.path.exists(tmp) else ''
        if code != 200 or '<html' in txt.lower()[:500]:
            txt = ''  # no robots file -> everything allowed
        rp.parse(txt.splitlines())
        _robots[host] = rp
    rp = _robots[host]
    return rp.can_fetch('*', url), (rp.crawl_delay('*') or 0)


def fetch(url, path, min_gap=3.0):
    """Download url to path unless cached. Returns path or None."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    host = _host(url)
    if os.path.exists(os.path.join(CACHE, f'BLOCKED_{host}.txt')):
        raise Blocked(host)
    ok, delay = _robots_ok(url)
    if not ok:
        print(f'robots.txt disallows {url} -> skipped', file=sys.stderr)
        return None
    gap = max(min_gap, float(delay) + 0.5 if delay else 0)
    w = gap - (time.time() - _last.get(host, 0))
    if w > 0:
        time.sleep(w)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    code = _curl(url, tmp)
    _last[host] = time.time()
    size = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    with open(os.path.join(CACHE, 'fetch_log.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(dict(url=url, status=code, bytes=size, time=time.strftime('%F %T'))) + '\n')
    if code == 429:
        open(os.path.join(CACHE, f'BLOCKED_{host}.txt'), 'w').write(time.strftime('%F %T') + ' HTTP 429 ' + url)
        os.remove(tmp)
        raise Blocked(host)
    if code != 200 or size == 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        return None
    os.replace(tmp, path)
    return path


# ================================================================== source files for a month
def get_danawa(month, nat, offline):
    p = os.path.join(CACHE, 'danawa', f'{nat}_{month}.html')
    if not os.path.exists(p) and not offline:
        try:
            fetch(DANAWA.format(nat=nat, m=month), p, min_gap=10.5)
        except Blocked:
            print('Danawa blocked (429) -> no Danawa values this run', file=sys.stderr)
    return p if os.path.exists(p) else None


def _hmc_candidates(month):
    """IR file names seen: 2022-sales-by-model, hmc-sales-by-model-december-y2023, -dec-y2025, -august-y2026."""
    y, m = int(month[:4]), int(month[5:])
    last = 12 if y < int(time.strftime('%Y')) else max(m, int(time.strftime('%m')) - 1)
    out = []
    for mm in range(last, m - 1, -1):
        full = MONTH_NAMES[mm - 1]
        for tag in dict.fromkeys([full, full[:3], 'sept' if mm == 9 else full]):
            out.append(f'{y}/hmc-sales-by-model-{tag}-y{y}.xlsx')
    return out


def get_hmc(month, offline, forced=None):
    """-> (DataFrame of the month or None, file name)"""
    y = int(month[:4])
    files = [forced] if forced else sorted(glob.glob(os.path.join(CACHE, 'hmc_ir', f'*{y}*.xlsx')), key=os.path.getmtime, reverse=True)
    for f in files:
        try:
            df = K.hmc_ir(f, y)
        except Exception as e:  # noqa: BLE001
            print(f'cannot read {f}: {e}', file=sys.stderr)
            continue
        if month in set(df.month):
            return df[df.month == month], os.path.basename(f)
    if offline or forced:
        return None, ''
    for rel in _hmc_candidates(month):
        p = os.path.join(CACHE, 'hmc_ir', rel.split('/')[-1])
        try:
            got = fetch(HMC_BASE + rel, p)
        except Blocked:
            return None, ''
        if got:
            df = K.hmc_ir(got, y)
            if month in set(df.month):
                return df[df.month == month], os.path.basename(got)
            return None, ''  # newest file exists but does not cover the month yet
    return None, ''


def get_kgm(month, forced=None):
    y = int(month[:4])
    files = [forced] if forced else sorted(glob.glob(os.path.join(CACHE, 'kgm_ir', f'*{y}*.xlsx')) +
                                           glob.glob(os.path.join(CACHE, 'kgm_ir', '*.xlsx')), key=os.path.getmtime, reverse=True)
    for f in dict.fromkeys(files):
        try:
            df = K.kgm_ir(f, y)
        except Exception:  # noqa: BLE001  (sheet for the year missing in an older file)
            continue
        if month in set(df.month):
            return df[df.month == month], os.path.basename(f)
    return None, ''


def get_press(month, maker, offline):
    y, m = month.split('-')
    slug = f"{'hyundai-motor-company' if maker == 'Hyundai' else 'kia'}-{MONTH_NAMES[int(m) - 1]}-{y}-sales"
    p = os.path.join(CACHE, 'press', slug + '.html')
    if not os.path.exists(p) and not offline:
        try:
            fetch('https://www.hyundaimotorgroup.com/ko/news/' + slug, p)
        except Blocked:
            pass
    if not os.path.exists(p):
        return None
    r = K.hmg_press(p)
    return r if r and r['month'] == month else None


def get_kaida(month, offline):
    """-> (brands {Brand: units}, total) from cache/kaida (article html or the audit history csv)."""
    for f in glob.glob(os.path.join(CACHE, 'kaida', f'kaida_{month}_*.html')):
        mo, b, t = K.kaida_press(f)
        if mo == month:
            return b, t
    hist = os.path.join(CACHE, 'kaida', 'kaida_brand_long.csv')
    if os.path.exists(hist):
        h = pd.read_csv(hist)
        h = h[h.period == month]
        if len(h):
            tot = h[h.brand == 'ALL_IMPORT'].units
            b = h[h.model_name_source == 'BRAND_TOTAL']
            return {K.KAIDA_BRAND.get(x, x): v for x, v in zip(b.brand, b.units)}, (float(tot.iloc[0]) if len(tot) else None)
    if offline:
        return None, None
    try:  # list page -> article for the month -> article page
        lst = fetch('https://www.kaida.co.kr/ko/kaida/bbsList.do?boardSeq=15&bbsType=kaida',
                    os.path.join(CACHE, 'kaida', f'list_{time.strftime("%Y%m%d")}.html'))
        if not lst:
            return None, None
        t = open(lst, encoding='utf-8', errors='ignore').read()
        mon = int(month[5:])
        for seq, no, title in re.findall(r"fn_egov_select\('(\d+)','(\d+)'\);\"[^>]*>\s*<span>([^<]*)</span>", t):
            if re.match(rf'{mon}월 수입 승용차', title.strip()):
                p = fetch(f'https://www.kaida.co.kr/ko/kaida/bbsView.do?type=kaida&articleSeq={seq}&boardSeq=15'
                          f'&bbsType=kaida&pageIndex=1&articleNo={no}', os.path.join(CACHE, 'kaida', f'kaida_{month}_{seq}.html'))
                if p:
                    mo, b, tot = K.kaida_press(p)
                    if mo == month:
                        return b, tot
    except Blocked:
        pass
    return None, None


# ================================================================== name variants
DOM_PRE = ['디 올 뉴 ', '더 올 뉴 ', '올 뉴 ', '더 뉴 ', '디 뉴 ', '뉴 ']
IMP_PRE = ['the all new ', 'the all-new ', 'all new ', 'all-new ', 'the new ', 'new ']


def norm(name, nat):
    s = name.strip().lower()
    for p in (DOM_PRE if nat == 'domestic' else IMP_PRE):
        if s.startswith(p):
            s = s[len(p):]
            break
    s = s.replace('-class', '') if nat != 'domestic' else s
    return re.sub(r'[\s\-]', '', s)


# ================================================================== main logic
def load_mapping():
    mp = pd.read_csv(os.path.join(HERE, 'mapping.csv'), dtype={'source_key': str}, keep_default_na=False)
    mp['priority'] = mp.priority.astype(int)
    return mp


def danawa_values(month, mp, offline, known_lines):
    """-> per-mapping-row values, unmapped lines, raw line tables {nat: df}"""
    tables, vals, unm = {}, {}, []
    for nat, sid in (('domestic', 'KR_DANAWA_DOMESTIC'), ('export', 'KR_DANAWA_IMPORT')):
        p = get_danawa(month, nat, offline)
        if not p:
            continue
        shown = K.danawa_month_shown(p)
        if shown and shown != month:
            print(f'Danawa {nat} page shows {shown}, not {month} -> ignored', file=sys.stderr)
            continue
        lines, _, brands = K.danawa_page(p, nat)
        tables[nat] = (lines, brands)
        rows = mp[mp.source_id == sid]
        # the same display name can appear twice (old and new generation, different model ids) -> add them
        line_val = lines.groupby(['brand_id', 'name']).sales.sum().to_dict()
        used = set()
        idx_norm = {}
        for i, r in rows.iterrows():
            parts = [tuple(x.split(':', 1)) for x in r.source_key.split('|') if x]
            v = sum(line_val.get(pt, 0.0) for pt in parts)
            used.update(pt for pt in parts if pt in line_val)
            listed = [pt[1] for pt in parts if pt in line_val]
            vals[i] = dict(value=v, lines=listed, variant=[])
            for b, n in parts:
                idx_norm.setdefault((b, norm(n, nat)), set()).add(i)
        for (b, n), v in line_val.items():
            if (b, n) in used:
                continue
            cand = idx_norm.get((b, norm(n, nat)), set())
            if len(cand) == 1:
                i = next(iter(cand))
                vals[i]['value'] += v
                vals[i]['variant'].append(n)
                continue
            unm.append(dict(Country='South Korea', Month=month, nat=nat, brand_id=b, brand_name=brands.get(b, ''), line=n,
                            sales=v, status='seen_before_not_ours' if (nat, n) in known_lines else 'NEW_LINE',
                            candidates=';'.join(f'{rows.loc[c, "Brand"]}/{rows.loc[c, "Model"]}' for c in cand)))
    return vals, pd.DataFrame(unm), tables


def collect(month, offline=True, hmc_file=None, kgm_file=None):
    """-> (std DataFrame, candidates DataFrame, unmapped DataFrame, checks DataFrame)"""
    mp = load_mapping()
    known = pd.read_csv(os.path.join(CACHE, 'audit_defs', 'dn_models.csv'), dtype=str)
    known_lines = set(zip(known.nat, known.name))
    for nat, k in zip(mp.source_id, mp.source_key):  # mapped names count as known
        if nat.startswith('KR_DANAWA'):
            for x in k.split('|'):
                known_lines.add(('domestic' if nat.endswith('DOMESTIC') else 'export', x.split(':', 1)[1]))
    files = {}
    dvals, unm, tables = danawa_values(month, mp, offline, known_lines)
    files['danawa'] = ','.join(sorted(tables))
    hmc, files['hmc'] = get_hmc(month, offline, hmc_file)
    kgm, files['kgm'] = get_kgm(month, kgm_file)
    press = {mk: get_press(month, mk, offline) for mk in ('Hyundai', 'Kia')}
    kaida_b, kaida_t = get_kaida(month, offline)

    cands = []
    for i, r in mp.iterrows():
        v, note, conf = None, r.note, r.confidence
        if r.source_id == 'KR_HMC_IR' and hmc is not None:
            names = r.source_key.split('|')
            sub = hmc[hmc.name.isin(names)]
            if len(sub):
                v = float(sub.units.sum())
                note = 'IR rows ' + '+'.join(f'{n}={u:g}' for n, u in zip(sub.name, sub.units) if u) + (f'; {note}' if note else '')
        elif r.source_id == 'KR_KGM_IR' and kgm is not None:
            sub = kgm[kgm.name.isin(r.source_key.split('|'))]
            if len(sub):
                v = float(sub.units.sum())
                note = 'KGM rows ' + '+'.join(f'{n}={u:g}' for n, u in zip(sub.name, sub.units) if u) + (f'; {note}' if note else '')
        elif r.source_id == 'KR_HMG_PRESS':
            mk, nm = r.source_key.split(':', 1)
            pr = press.get(mk)
            if pr and nm in pr['models']:
                v = pr['models'][nm]
        elif r.source_id.startswith('KR_DANAWA') and i in dvals:
            d = dvals[i]
            v = d['value']
            bits = []
            if d['lines']:
                bits.append('lines ' + '+'.join(d['lines']))
            if d['variant']:
                bits.append('NAME VARIANT added: ' + '+'.join(d['variant']) + ' (check mapping)')
                conf = 'low'
            if not d['lines'] and not d['variant']:
                bits.append('no mapped line listed (Danawa lists only lines with sales) -> 0')
            note = '; '.join(bits + ([note] if note else []))
        cands.append(dict(Brand=r.Brand, Model=r.Model, source_id=r.source_id, priority=r.priority, value=v,
                          confidence=conf, note=note, sales_last12=r.sales_last12))
    cand = pd.DataFrame(cands)
    cand.insert(0, 'Month', month)

    # brands with new unmapped import/domestic lines -> zeros of that brand are less certain
    new_brands = set(unm[unm.status == 'NEW_LINE'].brand_id) if len(unm) else set()
    bid_of = {}
    for r in mp[mp.source_id.str.startswith('KR_DANAWA')].itertuples():
        bid_of[(r.Brand, r.Model)] = r.source_key.split(':', 1)[0]

    std = []
    for (b, m), g in cand[cand.priority < 9].groupby(['Brand', 'Model'], sort=False):
        g = g.sort_values('priority')
        got = g[g.value.notna()]
        if not len(got):
            continue
        top = got.iloc[0]
        note, conf = top.note, top.confidence
        if top.priority > 1:
            miss = ','.join(g[(g.priority < top.priority) & g.value.isna()].source_id)
            note = f'fallback: {miss} not available for {month}; ' + note
            conf = 'medium' if conf == 'high' else conf
        if top.source_id.startswith('KR_DANAWA') and top.value == 0 and bid_of.get((b, m)) in new_brands:
            conf = 'low'
            note += '; brand has NEW unmapped Danawa lines this month (see unmapped file)'
        if month > LAST_ALLOWED:
            note = f'FORMAT CHECK ONLY (after {LAST_ALLOWED}); ' + note
        std.append(dict(Country='South Korea', Brand=b, Model=m, Month=month, Sales=float(top.value), value_type='reported',
                        estimated_share=0.0, source_id=top.source_id, confidence=conf, note=note))
    std = pd.DataFrame(std, columns=STD_COLS)
    checks = make_checks(month, std, cand, tables, hmc, kgm, press, kaida_b, kaida_t, mp)
    std.attrs['files'] = files
    return std, cand, unm, checks


def _chk(rows, check, key, expected, got, tol=0.02):
    if expected is None or got is None or pd.isna(expected) or pd.isna(got):
        status, ratio = 'warn', None
    else:
        ratio = (got / expected) if expected else (1.0 if got == 0 else None)
        status = 'ok' if ratio is not None and abs(ratio - 1) <= tol else 'warn'
    rows.append(dict(check=check, key=key, expected=expected, got=got, ratio=None if ratio is None else round(ratio, 4),
                     status=status))


IR_ALSO = {('Genesis', 'GV60'): [('Genesis', 'GV60 MAGMA')]}           # IR GV60 row = our GV60 + GV60 MAGMA
RELEASE_ALSO = {('Hyundai', 'Elantra'): [('Hyundai', 'Elantra N')]}    # release 아반떼 includes Avante N


def make_checks(month, std, cand, tables, hmc, kgm, press, kaida_b, kaida_t, mp):
    rows = []
    for nat in ('domestic', 'export'):
        _chk(rows, 'source_present', f'danawa_{nat}', 1, 1 if nat in tables else 0, 0)
    _chk(rows, 'source_present', 'hyundai_ir', 1, 1 if hmc is not None else 0, 0)
    _chk(rows, 'source_present', 'kgm_ir', 1, 1 if kgm is not None else 0, 0)
    for mk in ('Hyundai', 'Kia'):
        _chk(rows, 'source_present', f'press_{mk}', 1, 1 if press.get(mk) else 0, 0)
    _chk(rows, 'source_present', 'kaida_brand_totals', 1, 1 if kaida_b else 0, 0)
    imp_brands = set(mp[mp.source_id == 'KR_DANAWA_IMPORT'].Brand)
    std_b = std.groupby('Brand').Sales.sum()
    if 'export' in tables and kaida_t:
        lines, _ = tables['export']
        _chk(rows, 'kaida_total_vs_danawa_import_all_lines', 'ALL_IMPORT', kaida_t, float(lines.sales.sum()), 0.005)
        _chk(rows, 'kaida_total_vs_std_import_sum', 'ALL_IMPORT', kaida_t, float(std[std.Brand.isin(imp_brands)].Sales.sum()), 0.02)
    if kaida_b:
        for b, v in sorted(kaida_b.items(), key=lambda x: -x[1]):
            if b in imp_brands:
                _chk(rows, 'kaida_brand_vs_std_sum', b, v, float(std_b.get(b, 0.0)), 0.02 if v >= 200 else 0.10)
            else:
                rows.append(dict(check='kaida_brand_not_in_our_list', key=b, expected=v, got=None, ratio=None, status='ok'))
    if 'domestic' in tables:
        lines, _ = tables['domestic']
        bs = lines.groupby('brand_id').sales.sum()
        if kgm is not None:
            _chk(rows, 'kgm_ir_domestic_total_vs_danawa_kgm', 'KG Mobility', kgm.attrs['total'].get(month), float(bs.get('326', 0)), 0.01)
        if press.get('Kia') and press['Kia']['domestic_total']:
            _chk(rows, 'kia_release_domestic_vs_danawa_kia', 'Kia', press['Kia']['domestic_total'], float(bs.get('307', 0)), 0.01)
        if press.get('Hyundai') and press['Hyundai']['genesis_total']:
            _chk(rows, 'hyundai_release_genesis_total_vs_std', 'Genesis', press['Hyundai']['genesis_total'], float(std_b.get('Genesis', 0)), 0.005)
    # each primary vs Danawa and vs release figures
    piv = cand.pivot_table(index=['Brand', 'Model'], columns='source_id', values='value', aggfunc='first')
    for (b, m), r in piv.iterrows():
        s = std[(std.Brand == b) & (std.Model == m)]
        if not len(s):
            continue
        sv = float(s.Sales.iloc[0])
        if (b, m) == ('Hyundai', 'Sonata'):
            pass  # IR rows exclude the taxi since 2024; Danawa/release include it (known definition gap, not checked)
        elif 'KR_HMC_IR' in r and pd.notna(r.get('KR_HMC_IR')) and pd.notna(r.get('KR_DANAWA_DOMESTIC')):
            dn = r['KR_DANAWA_DOMESTIC'] + sum(piv.loc[x, 'KR_DANAWA_DOMESTIC'] for x in IR_ALSO.get((b, m), []) if x in piv.index)
            _chk(rows, 'hyundai_ir_vs_danawa', f'{b}|{m}', r['KR_HMC_IR'], dn, 0.02)
        if 'KR_KGM_IR' in r and pd.notna(r.get('KR_KGM_IR')) and pd.notna(r.get('KR_DANAWA_DOMESTIC')):
            _chk(rows, 'kgm_ir_vs_danawa', f'{b}|{m}', r['KR_KGM_IR'], r['KR_DANAWA_DOMESTIC'], 0.02)
        if 'KR_HMG_PRESS' in r and pd.notna(r.get('KR_HMG_PRESS')) and s.source_id.iloc[0] != 'KR_HMG_PRESS':
            sv += sum(float(std[(std.Brand == x[0]) & (std.Model == x[1])].Sales.sum()) for x in RELEASE_ALSO.get((b, m), []))
            _chk(rows, 'release_vs_std', f'{b}|{m}', r['KR_HMG_PRESS'], sv, 0.01)
    return pd.DataFrame(rows, columns=['check', 'key', 'expected', 'got', 'ratio', 'status'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', required=True)
    ap.add_argument('--offline', action='store_true')
    ap.add_argument('--hmc-file')
    ap.add_argument('--kgm-file')
    ap.add_argument('--no-write', action='store_true')
    ap.add_argument('--last-allowed', default=LAST_ALLOWED,
                    help='months after this are marked FORMAT CHECK ONLY (project rule today: 2026-07)')
    a = ap.parse_args()
    globals()['LAST_ALLOWED'] = a.last_allowed
    if not re.fullmatch(r'\d{4}-\d{2}', a.month):
        sys.exit('--month must be YYYY-MM')
    if a.month > LAST_ALLOWED:
        print(f'NOTE: {a.month} is after {LAST_ALLOWED}: format check only, not for our data', file=sys.stderr)
    std, cand, unm, checks = collect(a.month, a.offline, a.hmc_file, a.kgm_file)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f'std_korea_{a.month}.csv')
    if not a.no_write:
        std.to_csv(out, index=False, encoding='utf-8-sig')
        unm.to_csv(os.path.join(OUT, f'unmapped_korea_{a.month}.csv'), index=False, encoding='utf-8-sig')
        checks.to_csv(os.path.join(OUT, f'checks_korea_{a.month}.csv'), index=False, encoding='utf-8-sig')
    nser = cand[['Brand', 'Model']].drop_duplicates().shape[0]
    print(f'{a.month}: {len(std)}/{nser} series, sum {std.Sales.sum():,.0f}; by source '
          + ', '.join(f'{k} {v}' for k, v in std.source_id.value_counts().items())
          + f"; unmapped NEW lines {int((unm.status == 'NEW_LINE').sum()) if len(unm) else 0}"
          + f"; checks warn {int((checks.status == 'warn').sum())}/{len(checks)}; files {std.attrs['files']}", file=sys.stderr)
    print(os.path.relpath(out, ROOT).replace('\\', '/'))


if __name__ == '__main__':
    main()
