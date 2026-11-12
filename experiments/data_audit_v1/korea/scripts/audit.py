"""South Korea data audit: ours (D:/daizen/raw/data.csv, read-only) vs official/primary sources.
Writes only new files under D:/daizen/experiments/data_audit_v1/korea/.
"""
import re, os, json
import pandas as pd
import numpy as np
from hmc_defs import HMC_DEF, HMC_URL, HMC_BASE, HMC_EXCLUDED_NOTE
from dn_defs import DN_DEF, DN_BRAND

OUT = 'D:/daizen/experiments/data_audit_v1/korea/'
MAXP = '2026-07'
DN_URL = 'https://auto.danawa.com/auto/?Work=record&Tab=Model&Nation={nat}&Month={p}-00'

# ---------------- ours ----------------
d = pd.read_csv('D:/daizen/raw/data.csv')
k = d[(d.Country == 'South Korea') & (d.Month <= MAXP)].copy()
ours = k.set_index(['Brand', 'Model', 'Month']).Sales
series = k.groupby(['Brand', 'Model']).Sales.sum().sort_values(ascending=False)
DOMESTIC = {'Hyundai', 'Genesis', 'Kia', 'KG Mobility', 'Renault Korea'}
CHEV_DOM = {'Trax', 'TrailBlazer', 'Colorado', 'Traverse', 'Equinox', 'Tahoe'}  # sold by GM Korea, reported in its domestic sales

def is_domestic(b, m):
    return b in DOMESTIC or (b == 'Chevrolet' and m in CHEV_DOM)

# ---------------- sources ----------------
ir = pd.read_csv('hmc_long.csv')
ir = ir[ir.period <= MAXP]
irp = ir.pivot_table(index='period', columns='model_name_source', values='units', aggfunc='sum')

dn = pd.read_csv('dn_models.csv', dtype=str); dn['sales'] = dn.sales.astype(float)
dn = dn[dn.period <= MAXP]
tr = pd.read_csv('dn_trims.csv', dtype=str); tr['sales'] = tr.sales.astype(float)
tr = tr[tr.period <= MAXP]
dnd = dn[dn.nat == 'domestic']; dni = dn[dn.nat == 'export']
dvd = dnd.groupby(['period', 'name']).sales.sum().unstack()
dn_periods_dom = sorted(dnd.period.unique()); dn_periods_imp = sorted(dni.period.unique())

kb = pd.read_csv('kaida_brand_long.csv'); kbest = pd.read_csv('kaida_best_long.csv')
press = pd.read_csv('press_long.csv')
PRESS_MAP = {'그랜저': ('Hyundai', 'Grandeur'), '쏘나타': ('Hyundai', 'Sonata'), '아반떼': ('Hyundai', 'Elantra'), '팰리세이드': ('Hyundai', 'Palisade'),
             '싼타페': ('Hyundai', 'Santa Fe'), '투싼': ('Hyundai', 'Tucson'), '코나': ('Hyundai', 'Kona'), '포터': ('Hyundai', 'Porter II'),
             '스타리아': ('Hyundai', 'Staria'), '제네시스는 G80': ('Genesis', 'G80'), 'GV70': ('Genesis', 'GV70'), 'GV80': ('Genesis', 'GV80'),
             '쏘렌토로': ('Kia', 'Sorento'), '쏘렌토': ('Kia', 'Sorento'), '레이': ('Kia', 'Ray'), 'K5': ('Kia', 'K5'), 'K8': ('Kia', 'K8'),
             '셀토스로': ('Kia', 'Seltos'), '셀토스': ('Kia', 'Seltos'), '카니발': ('Kia', 'Carnival'), '스포티지': ('Kia', 'Sportage'),
             '모닝': ('Kia', 'Picanto'), '봉고Ⅲ': ('Kia', 'Bongo 3'), 'PV5': ('Kia', 'PV5'), '니로': ('Kia', 'Niro'), '아이오닉 5': ('Hyundai', 'Ioniq 5')}
PRESS_DEF_NOTE = {('Hyundai', 'Elantra'): 'press figure includes Avante N (ours excludes N; Elantra N separate)', ('Kia', 'Niro'): 'press figure includes Niro Plus (ours excludes it throughout)'}
press = press[press.model_name_source.isin(PRESS_MAP)].copy()
press['key'] = press.model_name_source.map(PRESS_MAP)
press_val = {(r.key, r.period): r.units for r in press.itertuples()}

# infer Danawa import brand ids from our import model names
def _inorm(s):
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    for pre in ['the all new ', 'the new ', 'all new ', 'all-new ', 'new ']:
        if s.startswith(pre): s = s[len(pre):]
    return s.replace('-class', '').replace(' ', '').replace('-', '')
_ourimp = k[~k.Brand.isin(['Hyundai', 'Genesis', 'Kia', 'KG Mobility', 'Renault Korea'])].drop_duplicates(['Brand', 'Model'])
_m2b = {}
for r in _ourimp.itertuples(): _m2b.setdefault(_inorm(r.Model), r.Brand)
_tmp = dn[dn.nat == 'export'].copy(); _tmp['ob'] = _tmp['name'].map(lambda x: _m2b.get(_inorm(x)))
for bid, g in _tmp.dropna(subset=['ob']).groupby('brand_id'):
    DN_BRAND.setdefault(bid, g.ob.value_counts().index[0])
# ---------------- official long ----------------
long_rows = []
for _, r in ir.iterrows():
    y = int(r.period[:4])
    long_rows.append(dict(period=r.period, source='Hyundai Motor IR Unit Sales by Model - Domestic (primary)', brand='Hyundai/Genesis',
                          model_name_source=f'{r.segment} | {r.model_name_source}', units=r.units, url_or_file=HMC_BASE + HMC_URL[y]))
for _, r in dn.iterrows():
    nat = 'domestic' if r.nat == 'domestic' else 'export'
    src = ('Danawa sales record - domestic makers (manufacturer-reported figures; secondary compilation)' if nat == 'domestic'
           else 'Danawa sales record - imports (KAIDA registration data; secondary compilation)')
    long_rows.append(dict(period=r.period, source=src, brand=DN_BRAND.get(r.brand_id, f'danawa_brand_{r.brand_id}'),
                          model_name_source=r['name'], units=r.sales, url_or_file=DN_URL.format(nat=nat, p=r.period)))
idmap = dn.drop_duplicates(['period', 'model_id']).set_index(['period', 'model_id'])['name']
bid_of = dn.drop_duplicates('model_id').set_index('model_id')['brand_id'].to_dict()
for _, r in tr.iterrows():
    nat = 'domestic' if r.nat == 'domestic' else 'export'
    nm = idmap.get((r.period, r.model_id), r.model_id)
    long_rows.append(dict(period=r.period, source='Danawa sales record - trim/powertrain breakdown (secondary)',
                          brand=DN_BRAND.get(bid_of.get(r.model_id, ''), f"danawa_brand_{bid_of.get(r.model_id, '')}"),
                          model_name_source=f'{nm} | {r.trim}', units=r.sales, url_or_file=DN_URL.format(nat=nat, p=r.period)))
for _, r in press.iterrows():
    long_rows.append(dict(period=r.period, source=r.source, brand=r.brand, model_name_source=r.model_name_source, units=r.units, url_or_file=r.url_or_file))
for df_ in (kb, kbest):
    for _, r in df_.iterrows():
        aid = re.search(r'(\d+)\.html', r.url_or_file).group(1)
        long_rows.append(dict(period=r.period, source=r.source, brand=r.brand if isinstance(r.brand, str) else '',
                              model_name_source=r.model_name_source, units=r.units,
                              url_or_file=f'https://www.kaida.co.kr/ko/news/newsView.do (article seq {aid}); cached {r.url_or_file}'))
off_long = pd.DataFrame(long_rows)
off_long.to_csv(OUT + 'korea_official_long.csv', index=False, encoding='utf-8-sig')

# ---------------- helpers ----------------
def ir_value(key, p):
    cols = [c for c in HMC_DEF[key] if c in irp.columns]
    if p not in irp.index: return np.nan
    v = irp.loc[p, cols]
    return v.sum(min_count=1)

def dn_cols(key, which=('base', 'opt')):
    pats = sum((DN_DEF[key][w] for w in which), [])
    return [c for c in dvd.columns if any(re.match(pt, c) for pt in pats)]

def dn_value(key, p, which=('base', 'opt')):
    if p not in dvd.index: return np.nan
    cols = dn_cols(key, which)
    v = dvd.loc[p, cols].sum(min_count=1) if cols else np.nan
    return 0.0 if pd.isna(v) else v   # Danawa lists only models with sales in the month -> absent = 0

# Official consistent definition for Danawa-based domestic series: which name groups count
DN_OFFICIAL_WHICH = {key: ('base', 'opt') for key in DN_DEF}
for key in [('Hyundai', 'Elantra'), ('Hyundai', 'Ioniq 5'), ('Hyundai', 'Casper'), ('Genesis', 'GV60'),
            ('Hyundai', 'County Electric'), ('KG Mobility', 'Rexton'), ('Hyundai', 'Staria'), ('Kia', 'K8'), ('KG Mobility', 'Tivoli'), ('Hyundai', 'Santa Fe')]:
    DN_OFFICIAL_WHICH[key] = ('base',)
# Musso: consistent family = Musso EV + all Musso-badged pickups (무쏘 스포츠/무쏘 칸 in 2025, 무쏘 from 2026)
DN_OFFICIAL_WHICH[('KG Mobility', 'Musso')] = ('base', 'opt')
# Niro: 니로 플러스 (taxi/PBV EV) excluded in ours throughout -> exclude
DN_DEF[('Kia', 'Niro')]['opt_official'] = [r'^니로 EV$', r'^니로 플러그인 하이브리드$']
DN_DEF[('Hyundai', 'Kona')]['opt'] = [r'^코나 electric$', r'^코나 일렉트릭$']  # Kona N (코나 N) not counted in ours

def dn_official(key, p):
    if key == ('Kia', 'Niro'):
        if p not in dvd.index: return np.nan
        cols = [c for c in dvd.columns if re.match(r'^니로$', c) or any(re.match(x, c) for x in DN_DEF[key]['opt_official'])]
        v = dvd.loc[p, cols].sum(min_count=1); return 0.0 if pd.isna(v) else v
    return dn_value(key, p, DN_OFFICIAL_WHICH[key])

# Import name normalisation (Danawa import names vs our model names)
def inorm(s):
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    for pre in ['the all new ', 'the new ', 'all new ', 'all-new ', 'new ']:
        if s.startswith(pre): s = s[len(pre):]
    s = s.replace('-class', '').replace(' ', '').replace('-', '')
    return s
dni = dni.copy(); dni['key'] = dni['name'].map(inorm)
dvi = dni.groupby(['period', 'key']).sales.sum().unstack()
dvi_names = dni.groupby('key')['name'].apply(lambda s: sorted(set(s)))
import imp_match
_di, _lines, _bid2b = imp_match.build(k, dn)
IMP = imp_match.infer(k, _di)
pd.to_pickle(IMP, 'imp_res.pkl')

# ---------------- audit ----------------
rows = []
months_all = sorted(k.Month.unique())
for (b, m), tot in series.items():
    key = (b, m)
    for p in months_all:
        o = ours.get((b, m, p), np.nan)
        if pd.isna(o): continue
        rec = dict(Brand=b, Model=m, Month=p, ours=o, official=np.nan, source='', official_ir=np.nan, official_press=press_val.get((key, p), np.nan), official_danawa=np.nan, note='')
        if key in HMC_DEF:
            rec['official_ir'] = ir_value(key, p)
        if key in DN_DEF and p in dn_periods_dom:
            rec['official_danawa'] = dn_official(key, p)
        if not is_domestic(b, m):
            if p in dn_periods_imp and key in IMP:
                rec['official_danawa'] = float(IMP[key]['official'].get(p, 0.0))
                ex = IMP[key]['expl'].get(p)
                if rec['official_danawa'] != o:
                    rec['note'] = ('ours = ' + ' + '.join(f'{n}x{mu}' if mu > 1 else n for n, mu in ex.items() if mu) if ex else 'not explained by Danawa lines')
                    rec['note'] += ' | definition lines: ' + ', '.join(IMP[key]['defn'])
        # choose primary comparison
        if key in HMC_DEF and not (key == ('Hyundai', 'Sonata') and p >= '2024-01') and not (key == ('Genesis', 'GV60') and p >= '2026-02'):
            rec['official'] = rec['official_ir']; rec['source'] = 'Hyundai IR (primary)'
        elif pd.notna(rec['official_press']) and key not in PRESS_DEF_NOTE:
            rec['official'] = rec['official_press']; rec['source'] = 'Manufacturer monthly press release (primary)'
        elif pd.notna(rec['official_danawa']):
            rec['official'] = rec['official_danawa']
            rec['source'] = 'Danawa domestic (manufacturer figures, secondary)' if is_domestic(b, m) else 'Danawa imports (KAIDA data, secondary)'
            if key == ('Hyundai', 'Sonata'): rec['note'] = 'IR excludes DN8 taxi from 2024; Danawa/press include it'
            if key == ('Genesis', 'GV60'): rec['note'] = 'IR does not split GV60 MAGMA; IR GV60 total = ours GV60 + GV60 MAGMA'
        rows.append(rec)
au = pd.DataFrame(rows)
au['diff'] = au.ours - au.official
def status(r):
    if pd.isna(r.official): return 'NO_OFFICIAL'
    if r['diff'] == 0: return 'EXACT'
    tol = max(5.0, 0.02 * abs(r.official))
    if abs(r['diff']) <= tol: return 'MINOR_DIFF'
    return 'DIFF'
au['status'] = au.apply(status, axis=1)
au.to_pickle('audit_tmp.pkl')
pd.to_pickle({k_: dn_cols(k_, DN_OFFICIAL_WHICH[k_]) if k_ != ('Kia', 'Niro') else ['니로', '니로 EV', '니로 플러그인 하이브리드'] for k_ in DN_DEF}, 'dn_official_cols.pkl')
print(au.status.value_counts())
