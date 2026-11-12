"""Build korea_audit.csv, korea_mapping.csv, korea_audit_summary.csv, korea_issue_list.csv from audit_tmp.pkl."""
import re
import pandas as pd, numpy as np
from hmc_defs import HMC_DEF, HMC_EXCLUDED_NOTE
from dn_defs import DN_DEF

OUT = 'D:/daizen/experiments/data_audit_v1/korea/'
au = pd.read_pickle('audit_tmp.pkl')
ir = pd.read_csv('hmc_long.csv'); irp = ir.pivot_table(index='period', columns='model_name_source', values='units', aggfunc='sum')
dn = pd.read_csv('dn_models.csv', dtype=str); dn['sales'] = dn.sales.astype(float)
dnd = dn[dn.nat == 'domestic']; dvd = dnd.groupby(['period', 'name']).sales.sum().unstack()
tr = pd.read_csv('dn_trims.csv', dtype=str); tr['sales'] = tr.sales.astype(float)
press = pd.read_csv('press_long.csv')

# --- BMW 2 Series / Gran Coupe: make the official definitions non-overlapping (GC trims belong to GC only)
_dni = dn[dn.nat == 'export']; _tri = tr[tr.nat == 'export']
def _status(o, off):
    if pd.isna(off): return 'NO_OFFICIAL'
    if o - off == 0: return 'EXACT'
    return 'MINOR_DIFF' if abs(o - off) <= max(5.0, 0.02 * abs(off)) else 'DIFF'
for idx, r in au[(au.Brand == 'BMW') & au.Model.isin(['2 Series', '2 Series Gran Coupe'])].iterrows():
    p = r.Month
    ids = _dni[(_dni.period == p) & (_dni['name'] == '2 Series')].model_id
    line = _dni[(_dni.period == p) & (_dni['name'] == '2 Series')].sales.sum()
    gct = _tri[(_tri.period == p) & _tri.model_id.isin(ids) & _tri.trim.str.contains('Gran Coupe')].sales.sum()
    gcl = _dni[(_dni.period == p) & (_dni['name'] == '2 Series Gran Coupe')].sales.sum()
    if p >= '2025-06': continue  # separate GC line from 2025-06; definitions already disjoint
    off = (line - gct) if r.Model == '2 Series' else (gct + gcl)
    au.at[idx, 'official'] = off; au.at[idx, 'official_danawa'] = off; au.at[idx, 'diff'] = r.ours - off
    au.at[idx, 'status'] = _status(r.ours, off)
    au.at[idx, 'note'] = f"official redefined as non-overlapping: '2 Series' line {line:.0f} contains Gran Coupe trims {gct:.0f}" + (' (see KR-12)' if gct else '')
IRURL = 'Hyundai IR xlsx hmc-sales-by-model ({y}) https://www.hyundai.com/worldwide/en/company/ir/ir-resources/sales-results'
DNURL = 'https://auto.danawa.com/auto/?Work=record&Tab=Model&Nation={n}&Month={p}-00'

def a(b, m, p, col):
    r = au[(au.Brand == b) & (au.Model == m) & (au.Month == p)]
    return r[col].iloc[0] if len(r) else np.nan

def trim_val(p, model_name, trim_pat):
    ids = dnd[(dnd.period == p) & (dnd.name == model_name)].model_id
    t = tr[(tr.period == p) & (tr.model_id.isin(ids)) & tr.trim.str.contains(trim_pat)]
    return t.sales.sum() if len(t) else 0.0

issues = []
def add(iid, conf, typ, members, p, ours, official, comps, desc, ev):
    issues.append(dict(issue_id=iid, confidence=conf, type=typ, members=members, Month=p, ours=ours, official=official,
                       components=comps, description=desc, evidence=ev))

# KR-01 Kona EV double count
for p in au[(au.Model == 'Kona')].Month:
    if not ('2019-04' <= p <= '2021-04'): continue
    o = a('Hyundai', 'Kona', p, 'ours'); off = a('Hyundai', 'Kona', p, 'official_ir')
    ev = sum(irp.loc[p, c] for c in ['Kona (EV)', 'Kona (OS EV)'] if c in irp.columns and pd.notna(irp.loc[p, c]))
    ice = off - ev
    if o - off == ev and ev > 0:
        add('KR-01', 'CONFIRMED', 'DOUBLE_COUNT', 'Kona', p, o, off, f'IR Kona ICE+HEV {ice:.0f} + IR Kona EV {ev:.0f} = {off:.0f}; ours = official + EV {ev:.0f}',
            'Kona EV counted twice: ours = Kona family total (which already contains EV) + EV again. Pattern holds every month 2019-04..2021-03.',
            f"{IRURL.format(y=p[:4])} rows Kona/Kona (EV)/Kona (HEV) [OS from 2021]; Danawa '코나' row {a('Hyundai','Kona',p,'official_danawa'):.0f} incl. trim '전기' {ev:.0f} ({DNURL.format(n='domestic', p=p)})")
    elif o != off:
        add('KR-02', 'CONFIRMED', 'MISMATCH_OTHER', 'Kona', p, o, off, f'IR Kona OS {irp.loc[p,"Kona (OS)"]:.0f} + OS HEV {irp.loc[p,"Kona (OS HEV)"]:.0f} + OS EV {irp.loc[p,"Kona (OS EV)"]:.0f}',
            f'Excess {o-off:.0f} not equal to EV ({ev:.0f}); probably the tail of KR-01 with a different EV figure. IR and Danawa agree on {off:.0f}.',
            f"{IRURL.format(y=p[:4])}; {DNURL.format(n='domestic', p=p)} ('코나' {a('Hyundai','Kona',p,'official_danawa'):.0f})")

# KR-03 Niro EV double count (Danawa '니로' row already contains EV trim)
for p in sorted(au[(au.Model == 'Niro')].Month):
    o = a('Kia', 'Niro', p, 'ours'); off = a('Kia', 'Niro', p, 'official_danawa')
    if pd.isna(off) or o == off: continue
    evt = trim_val(p, '니로', '전기')
    if evt > 0 and o - off == evt:
        add('KR-03', 'CONFIRMED', 'DOUBLE_COUNT', 'Niro', p, o, off, f"Danawa '니로' row {dvd.loc[p,'니로']:.0f} (contains trim 전기 {evt:.0f}) + other Niro rows; ours = official + 전기 {evt:.0f}",
            'Niro EV counted twice: the Niro model line already contains the EV trim, and the EV trim was added again. Kia publishes no model-level IR file, so the check is arithmetic on the manufacturer-reported line (same mechanism as KR-01, which is confirmed by Hyundai IR).',
            DNURL.format(n='domestic', p=p))
    elif abs(o - off) > max(5, 0.02 * off):
        add('KR-03b', 'UNCERTAIN', 'MISMATCH_OTHER', 'Niro', p, o, off, '', 'Niro differs from Danawa line and not explained by EV trim', DNURL.format(n='domestic', p=p))

# KR-04 Ioniq 5 / Ioniq 5 N split vs IR
for p in sorted(au[(au.Model == 'Ioniq 5 N')].Month):
    o5 = a('Hyundai', 'Ioniq 5', p, 'ours'); oN = a('Hyundai', 'Ioniq 5 N', p, 'ours')
    i5 = a('Hyundai', 'Ioniq 5', p, 'official_ir'); iN = a('Hyundai', 'Ioniq 5 N', p, 'official_ir')
    if pd.notna(i5) and (o5 != i5 or oN != iN) and abs(oN - iN) > 2:
        add('KR-04', 'CONFIRMED', 'MISMATCH_OTHER', 'Ioniq 5; Ioniq 5 N', p, f'{o5:.0f}; {oN:.0f}', f'{i5:.0f}; {iN:.0f}',
            f'IR IONIQ 5 (NE) {i5:.0f}, IONIQ 5 N (NE N) {iN:.0f}; sum ours {o5+oN:.0f} vs IR {i5+iN:.0f}',
            'Units shifted between Ioniq 5 and Ioniq 5 N relative to Hyundai IR; the two-series total matches (or nearly). Our values equal Danawa, which splits N differently from the IR file.',
            IRURL.format(y=p[:4]) + '; ' + DNURL.format(n='domestic', p=p))

# KR-05 Bongo 3 definition change at the end of the series
for p in ['2026-05', '2026-06', '2026-07']:
    o = a('Kia', 'Bongo 3', p, 'ours'); pr = a('Kia', 'Bongo 3', p, 'official_press'); dv = a('Kia', 'Bongo 3', p, 'official_danawa')
    sp = trim_val(p, '봉고 3', '특장')
    add('KR-05', 'CONFIRMED', 'DEFINITION_CHANGE', 'Bongo 3', p, o, pr,
        f"Kia press 봉고Ⅲ {pr:.0f}; Danawa 봉고 3 + 봉고 3 EV = {dv:.0f} (특장차 trim {sp:.0f} - absent from 2026-05)",
        ('Ours follows the Danawa line, which stopped including the special-body (특장차) trim from 2026-05, so 2026-05/06 are understated vs the Kia release.' if p != '2026-07'
         else 'Ours switches to the Kia press-release figure in 2026-07 (2,846) while 2026-05/06 use the narrower Danawa line, creating a level jump (+{:.0f} vs Danawa definition).'.format(o - dv)),
        f"Kia monthly release {press[(press.period==p)&(press.model_name_source=='봉고Ⅲ')].url_or_file.iloc[0]}; {DNURL.format(n='domestic', p=p)}")
# context: 2023 Kia press vs ours (Danawa incl. special trims) -> ours slightly above press
for p in ['2023-01', '2023-02', '2023-05', '2023-07', '2023-10']:
    o = a('Kia', 'Bongo 3', p, 'ours'); pr = a('Kia', 'Bongo 3', p, 'official_press')
    if pd.notna(pr) and o != pr:
        add('KR-05b', 'UNCERTAIN', 'DEFINITION_CHANGE', 'Bongo 3', p, o, pr, f"Kia press 봉고Ⅲ {pr:.0f}; Danawa 봉고 3 + EV {a('Kia','Bongo 3',p,'official_danawa'):.0f}",
            'Before 2026 ours (Danawa line incl. EV and 특장차) is slightly above the Kia press figure; the exact press definition is not stated.',
            f"{press[(press.period==p)&(press.model_name_source=='봉고Ⅲ')].url_or_file.iloc[0]}")

# KR-06 Musso definition change
for p in sorted(au[(au.Model == 'Musso')].Month):
    o = a('KG Mobility', 'Musso', p, 'ours'); off = a('KG Mobility', 'Musso', p, 'official_danawa')
    if pd.isna(off): continue
    row = dvd.loc[p] if p in dvd.index else pd.Series(dtype=float)
    comps = '; '.join(f'{n} {row[n]:.0f}' for n in ['무쏘', '무쏘 EV', '무쏘 스포츠', '무쏘 칸'] if n in row.index and pd.notna(row[n]))
    if o != off:
        add('KR-06', 'CONFIRMED', 'DEFINITION_CHANGE', 'Musso', p, o, off, comps,
            'In 2025 ours = Musso EV only (the ICE Musso Sports / Musso Khan pickups, formerly Rexton Sports, are not in any of our series); from 2026-01 ours = Musso EV + the new ICE Musso pickup. The jump at 2026-01 is partly a scope change.',
            DNURL.format(n='domestic', p=p))
    elif p >= '2026-01':
        add('KR-06', 'CONFIRMED', 'DEFINITION_CHANGE', 'Musso', p, o, off, comps, 'From 2026-01 ours includes the ICE Musso pickup (matches family total).', DNURL.format(n='domestic', p=p))

# KR-07 Tivoli Air
for p in sorted(au[(au.Model == 'Tivoli')].Month):
    if p in dvd.index and '티볼리 에어' in dvd.columns and pd.notna(dvd.loc[p, '티볼리 에어']):
        o = a('KG Mobility', 'Tivoli', p, 'ours')
        add('KR-07', 'UNCERTAIN', 'DEFINITION_CHANGE', 'Tivoli', p, o, o + dvd.loc[p, '티볼리 에어'], f"티볼리 {dvd.loc[p,'티볼리']:.0f} + 티볼리 에어 {dvd.loc[p,'티볼리 에어']:.0f}",
            'Tivoli Air (long-body Tivoli) excluded while it is reported as its own line (2020-10..2025-02). Whether Tivoli Air was inside the Tivoli figure before 2020-10 could not be verified from the sources used.',
            DNURL.format(n='domestic', p=p))

# Single-month / older conflicts vs Hyundai IR where ours = Danawa
singles = [('Genesis', 'G80', '2016-07', 'CONFIRMED', 'SPIKE', 'Ours exceeds both Hyundai IR (G80 row 4,469) and Danawa (G80 4,574, Genesis DH not listed). Source of the extra units unknown.'),
           ('Hyundai', 'Tucson', '2018-08', 'UNCERTAIN', 'MISMATCH_OTHER', "Ours = Danawa 투싼 4,808 + 투싼(2017) 230; IR Tucson (TL) 4,148. The gap 890 equals Danawa's '디젤 1.6' trim of the new Tucson line; possible duplicate in Danawa but not provable."),
           ('Genesis', 'G70', '2017-09', 'UNCERTAIN', 'MISMATCH_OTHER', 'Launch month; ours = Danawa 386, IR 199.'),
           ('Genesis', 'G90', '2024-03', 'UNCERTAIN', 'MISMATCH_OTHER', 'Ours = Danawa 550; IR G90 (RS4) 715.'),
           ('Hyundai', 'Grandeur', '2016-11', 'UNCERTAIN', 'MISMATCH_OTHER', 'Model-change month (HG->IG); ours = Danawa 7,984, IR 7,747.')]
for b, m, p, conf, typ, desc in singles:
    add('KR-08' if m == 'G80' else 'KR-09', conf, typ, m, p, a(b, m, p, 'ours'), a(b, m, p, 'official_ir'), f"IR {a(b,m,p,'official_ir'):.0f}; Danawa {a(b,m,p,'official_danawa'):.0f}", desc,
        IRURL.format(y=p[:4]) + '; ' + DNURL.format(n='domestic', p=p))
for p in ['2016-02', '2016-03', '2016-04', '2016-05']:
    add('KR-10', 'UNCERTAIN', 'DEFINITION_CHANGE', 'Sonata', p, a('Hyundai', 'Sonata', p, 'ours'), a('Hyundai', 'Sonata', p, 'official_ir'),
        f"IR YF {irp.loc[p,'Sonata (YF)']:.0f} + LF {irp.loc[p,'Sonata (LF)']:.0f} + HEV {irp.loc[p,'Sonata (HEV)']:.0f} + PHEV {irp.loc[p,'Sonata (PHEV)']:.0f}",
        'Old-generation YF Sonata (taxi) units missing in 2016-02..04 (Danawa lumped them into a bus/truck line); 2016-05 ours above IR LF by 217.',
        IRURL.format(y='2016') + '; ' + DNURL.format(n='domestic', p=p))
# Source-definition note (not an error in ours)
for p in sorted(au[(au.Model == 'Sonata') & (au.Month >= '2024-01')].Month):
    o = a('Hyundai', 'Sonata', p, 'ours'); i = a('Hyundai', 'Sonata', p, 'official_ir')
    t = trim_val(p, '쏘나타 디 엣지', '택시') + trim_val(p, '쏘나타 뉴 라이즈', '택시')
    if o != i and t == 0:
        add('KR-09', 'UNCERTAIN', 'MISMATCH_OTHER', 'Sonata', p, o, i, f'IR DN8+DN8 HEV {i:.0f}; Danawa line {a("Hyundai","Sonata",p,"official_danawa"):.0f} (no taxi trim listed)',
            'Model-change month (Sonata facelift); ours = Danawa, IR lower by the stated amount.', IRURL.format(y=p[:4]) + '; ' + DNURL.format(n='domestic', p=p))
    elif o != i:
        add('KR-11', 'CONFIRMED', 'SOURCE_DEFINITION_NOTE', 'Sonata', p, o, a('Hyundai', 'Sonata', p, 'official'),
            f'IR DN8+DN8 HEV {i:.0f}; Danawa taxi trim {t:.0f}; IR + taxi = {i+t:.0f}',
            'Not an error in ours: from 2024 the Hyundai IR file leaves the Sonata taxi out of the Sonata rows; our series (= press release / Danawa) keeps it, consistent with earlier years when IR included the LF taxi.',
            IRURL.format(y=p[:4]) + '; ' + DNURL.format(n='domestic', p=p))

iss = pd.DataFrame(issues)
iss.to_pickle('issues_dom.pkl')
print(iss.groupby(['issue_id', 'confidence', 'type']).size())

# ================= imports =================
IMP = pd.read_pickle('imp_res.pkl')
kb = pd.read_csv('kaida_brand_long.csv'); kb = kb[kb.model_name_source == 'BRAND_TOTAL']
d0 = pd.read_csv('D:/daizen/raw/data.csv'); kk = d0[(d0.Country == 'South Korea') & (d0.Month <= '2026-07')]
dni = dn[dn.nat == 'export']; tri = tr[tr.nat == 'export']
def kaida(brand, p):
    r = kb[(kb.brand.replace({'MINI': 'Mini'}) == brand) & (kb.period == p)]
    return r.units.iloc[0] if len(r) else np.nan
def oursum(brand, p): return kk[(kk.Brand == brand) & (kk.Month == p)].Sales.sum()
KURL = 'KAIDA monthly release https://www.kaida.co.kr/ko/news/newsView.do (seq {s})'
kseq = {r.period: re.search(r'(\d+)\.html', r.url_or_file).group(1) for r in kb.itertuples()}
# KR-12 BMW 2 Series Gran Coupe inside 2 Series line
for p in sorted(au[(au.Model == '2 Series Gran Coupe')].Month):
    if p >= '2025-06': continue
    gc = a('BMW', '2 Series Gran Coupe', p, 'ours'); s2 = a('BMW', '2 Series', p, 'ours')
    if not gc or pd.isna(gc): continue
    ids = dni[(dni.period == p) & (dni['name'] == '2 Series')].model_id
    t = tri[(tri.period == p) & tri.model_id.isin(ids)]
    gct = t[t.trim.str.contains('Gran Coupe')].sales.sum(); line = dni[(dni.period == p) & (dni['name'] == '2 Series')].sales.sum()
    kt = kaida('BMW', p); osb = oursum('BMW', p)
    add('KR-12', 'CONFIRMED', 'DOUBLE_COUNT', '2 Series; 2 Series Gran Coupe', p, f'{s2:.0f} + {gc:.0f} = {s2+gc:.0f}', line,
        f"Danawa/KAIDA '2 Series' line {line:.0f} incl. Gran Coupe trims {gct:.0f}; our 2 Series = whole line, our 2 Series Gran Coupe = the GC trims again"
        + (f"; KAIDA BMW brand total {kt:.0f} vs our BMW sum {osb:.0f}" if pd.notna(kt) else ''),
        'Gran Coupe registrations counted twice (inside 2 Series and again as 2 Series Gran Coupe) until 2025-05; from 2025-06 the source lists Gran Coupe as its own line and the overlap stops.',
        DNURL.format(n='export', p=p) + (('; ' + KURL.format(s=kseq[p])) if p in kseq else ''))
# KR-13 Ferrari 12Cilindri duplicate of 12Cilindri Spider
p = '2026-07'
add('KR-13', 'CONFIRMED', 'DOUBLE_COUNT', '12Cilindri; 12Cilindri Spider', p, f"{a('Ferrari','12Cilindri',p,'ours'):.0f} + {a('Ferrari','12Cilindri Spider',p,'ours'):.0f}",
    f"0 + {a('Ferrari','12Cilindri Spider',p,'official'):.0f}", f"KAIDA Ferrari brand total {kaida('Ferrari',p):.0f} vs our Ferrari sum {oursum('Ferrari',p):.0f}; Danawa lists only 12Cilindri Spider (5)",
    '12Cilindri (coupe) 2026-07 value equals the Spider value and has no source line; our Ferrari total exceeds the official KAIDA brand total by exactly that amount.',
    KURL.format(s=kseq[p]) + '; ' + DNURL.format(n='export', p=p))

iss = pd.DataFrame(issues)
iss.to_csv(OUT + 'korea_issue_list.csv', index=False, encoding='utf-8-sig')

# ================= audit csv =================
aud = au[['Brand', 'Model', 'Month', 'ours', 'official', 'diff', 'status', 'source', 'official_ir', 'official_press', 'official_danawa', 'note']].copy()
aud.to_csv(OUT + 'korea_audit.csv', index=False, encoding='utf-8-sig')

# ================= mapping =================
from dn_defs import DN_DEF as DD
DNC = pd.read_pickle('dn_official_cols.pkl')
mp = []
for (b, m), g in au.groupby(['Brand', 'Model'], sort=False):
    key = (b, m)
    srcs = g.source.value_counts().to_dict()
    row = dict(Brand=b, Model=m, market='import (KAIDA registrations)' if 'imports' in ' '.join(srcs) else 'domestic maker (sales)',
               primary_source='', primary_components='', secondary_source='', secondary_components='', excluded_or_notes='')
    if key in HMC_DEF:
        row['primary_source'] = 'Hyundai IR Unit Sales by Model (Domestic)'
        row['primary_components'] = ' + '.join(HMC_DEF[key])
        row['excluded_or_notes'] = HMC_EXCLUDED_NOTE.get(key, '')
    if any('press' in s for s in srcs):
        row['primary_source'] = (row['primary_source'] + '; ' if row['primary_source'] else '') + 'Hyundai/Kia monthly press release text (months available: ' + ','.join(sorted(g[g.official_press.notna()].Month)) + ')'
    if key in DD:
        row['secondary_source'] = 'Danawa domestic model lines (manufacturer figures)'
        row['secondary_components'] = ' + '.join(DNC[key])
        allc = set(c for c in dvd.columns if any(re.match(pt, c) for pt in DD[key]['base'] + DD[key]['opt']))
        notin = sorted(allc - set(DNC[key]))
        if notin: row['excluded_or_notes'] = (row['excluded_or_notes'] + '; ' if row['excluded_or_notes'] else '') + 'Danawa lines not counted: ' + ', '.join(notin)
    elif key in IMP:
        row['primary_source'] = (row['primary_source'] + '; ' if row['primary_source'] else '') + 'KAIDA brand totals (check of brand sum only)'
        row['secondary_source'] = 'Danawa import model lines (KAIDA registration data)'
        row['secondary_components'] = ' + '.join(IMP[key]['defn']) + ('' if set(IMP[key]['defn']) == set(IMP[key]['lines']) else ' (not included: ' + ', '.join(sorted(set(IMP[key]['lines']) - set(IMP[key]['defn']))) + ')')
    mp.append(row)
mpd = pd.DataFrame(mp)

EXTRA_NOTES = {
 ('Hyundai','Sonata'): 'From 2024-01 the IR file excludes the Sonata taxi; audit uses press/Danawa (incl. taxi) from 2024-01.',
 ('Genesis','GV60'): 'IR has no MAGMA split; from 2026-02 audit uses Danawa (IR GV60 = ours GV60 + GV60 MAGMA).',
 ('Hyundai','Inster'): 'Inster = Casper Electric (domestic name).',
 ('Kia','Picanto'): 'Picanto = Morning (domestic name).', ('Kia','K900'): 'K900 = K9 (domestic name).',
 ('Renault Korea','Arkana'): 'Arkana = XM3 until 2024-03 (renamed).', ('Kia','Niro'): 'Niro Plus (taxi/PBV EV) not counted; Kia press figure includes it.',
 ('Kia','Bongo 3'): 'Includes Bongo III EV; see KR-05 for the 2026-05..07 definition break.',
 ('KG Mobility','Musso'): 'See KR-06: Musso Sports/Khan pickups (ex Rexton Sports) missing in 2025, new Musso pickup included from 2026-01.',
 ('KG Mobility','Rexton'): 'Rexton SUV only (Rexton Sports pickups not in any of our series). Our series starts 2017-05 (G4 Rexton); predecessor Rexton W not included.',
 ('Hyundai','Staria'): 'Includes Staria Electric (2026-06+); predecessor Grand Starex not included (series starts 2021-04).',
 ('Chevrolet','Trax'): 'Trax (to 2023-04) + Trax Crossover (from 2023-04).',
}
for i, r in mpd.iterrows():
    n = EXTRA_NOTES.get((r.Brand, r.Model))
    if n: mpd.at[i, 'excluded_or_notes'] = (r.excluded_or_notes + '; ' if r.excluded_or_notes else '') + n
mpd.to_csv(OUT + 'korea_mapping.csv', index=False, encoding='utf-8-sig')

# ================= summary =================
SEV = {'DOUBLE_COUNT': 4, 'DEFINITION_CHANGE': 3, 'SPIKE': 2, 'MISMATCH_OTHER': 1}
memb = {}
for r in iss.itertuples():
    if r.type not in SEV: continue
    for mname in [x.strip() for x in r.members.split(';')]:
        memb.setdefault(mname, []).append((r.issue_id, r.type, r.confidence, r.Month))
tot_vol = au.ours.sum()
rows = []
for (b, m), g in au.groupby(['Brand', 'Model'], sort=False):
    cmp_ = g[g.status != 'NO_OFFICIAL']
    g36 = cmp_[(cmp_.Month >= '2023-08')]
    prim = cmp_[cmp_.source.str.contains('primary')]
    iss_m = [x for x in memb.get(m, []) if (b, m) in set(zip(au.Brand, au.Model))]
    # brand check for members names shared across brands (none in Korea except generic)
    cat, conf = 'OK', ''
    if len(cmp_) == 0: cat = 'UNMATCHED'
    if iss_m:
        top = max(iss_m, key=lambda x: (x[2] == 'CONFIRMED', SEV[x[1]]))
        cat = 'MISMATCH_OTHER' if top[1] == 'SPIKE' else top[1]; conf = top[2]
    dmon = set(cmp_[cmp_.status == 'DIFF'].Month); covered = set(x[3] for x in iss_m)
    if (dmon - covered) and cat == 'OK': cat, conf = 'MISMATCH_OTHER', 'UNCERTAIN'
    rows.append(dict(Brand=b, Model=m, ours_total_2016_2026_07=g.ours.sum(), share_of_korea_volume_pct=round(100 * g.ours.sum() / tot_vol, 3),
                     months_in_ours=len(g), months_compared=len(cmp_), months_vs_primary=len(prim), exact=int((cmp_.status == 'EXACT').sum()),
                     minor_diff=int((cmp_.status == 'MINOR_DIFF').sum()), differing=int((cmp_.status == 'DIFF').sum()),
                     net_diff=cmp_['diff'].sum(), abs_diff=cmp_['diff'].abs().sum(),
                     months_compared_2023_08_2026_07=len(g36), differing_2023_08_2026_07=int((g36.status == 'DIFF').sum()), net_diff_2023_08_2026_07=g36['diff'].sum(),
                     category=cat, category_confidence=conf, issue_ids=';'.join(sorted(set(x[0] for x in iss_m))),
                     sources_used='; '.join(f'{k_} ({v})' for k_, v in cmp_.source.value_counts().items())))
sm = pd.DataFrame(rows).sort_values('ours_total_2016_2026_07', ascending=False)
sm.to_csv(OUT + 'korea_audit_summary.csv', index=False, encoding='utf-8-sig')
print(sm.category.value_counts()); print(iss.groupby(['issue_id','confidence','type']).size())
