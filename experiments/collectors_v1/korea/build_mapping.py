"""Build korea/mapping.csv (one row per our series x source) from the definitions that were checked month by month
in experiments/data_audit_v1/korea (hmc_defs.py, dn_defs.py -> dn_official_cols, imp_match -> imp_res) plus KGM IR
and the HMG press names. Run once; the collector reads only mapping.csv.

priority: 1 = primary, 2 = fallback when the primary has no value for the month, 9 = check only (never written).
source_key:
  KR_HMC_IR            IR row names joined by '|'
  KR_KGM_IR            KGM IR Domestic row names joined by '|'
  KR_HMG_PRESS         maker:name in the 국내 paragraph of the HMG newsroom release
  KR_DANAWA_DOMESTIC   brand_id:line name joined by '|'   (line absent on the page = 0 sales)
  KR_DANAWA_IMPORT     brand_id:line name joined by '|'
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
AUDIT = os.path.join(ROOT, 'experiments', 'data_audit_v1', 'korea')
sys.path.insert(0, os.path.join(AUDIT, 'scripts'))
from hmc_defs import HMC_DEF, HMC_EXCLUDED_NOTE  # noqa: E402

# audit work files (session scratchpad of the audit; copied into korea/cache/audit_defs on first run)
DEFS = os.path.join(HERE, 'cache', 'audit_defs')
SCRATCH = os.environ.get('KR_AUDIT_WORK', '')


def load_defs():
    os.makedirs(DEFS, exist_ok=True)
    for f in ['dn_official_cols.pkl', 'imp_res.pkl', 'dn_models.csv']:
        dst = os.path.join(DEFS, f)
        if not os.path.exists(dst):
            src = os.path.join(SCRATCH, f)
            if not os.path.exists(src):
                sys.exit(f'missing {dst}; set KR_AUDIT_WORK to the audit work folder that holds {f}')
            import shutil
            shutil.copy(src, dst)
    dnc = pd.read_pickle(os.path.join(DEFS, 'dn_official_cols.pkl'))
    imp = pd.read_pickle(os.path.join(DEFS, 'imp_res.pkl'))
    dn = pd.read_csv(os.path.join(DEFS, 'dn_models.csv'), dtype=str)
    return dnc, imp, dn


KGM = {  # our series -> KGM IR Domestic rows (ICE + HEV/EV rows are added)
    'Tivoli': ['Tivoli'], 'Torres': ['Torres', 'Torres HEV'], 'Torres EVX': ['Torres EVX'], 'Rexton': ['Rexton'],
    'Musso': ['Musso', 'Musso EV'], 'Actyon': ['Actyon', 'Actyon HEV'],
}
KGM_NOTE = {
    'Musso': "2025 sheet rows 'Musso sports'/'Musso Khan' (ex Rexton Sports pickups) are NOT in our series (KR-06); "
             "from 2026 the ICE pickup row is 'Musso' and is included",
    'Tivoli': "'Korando'/'Korando EV' rows are not our series",
}
PRESS = {  # audit PRESS_MAP (name in the 국내 paragraph)
    ('Hyundai', 'Grandeur'): '그랜저', ('Hyundai', 'Sonata'): '쏘나타', ('Hyundai', 'Elantra'): '아반떼',
    ('Hyundai', 'Palisade'): '팰리세이드', ('Hyundai', 'Santa Fe'): '싼타페', ('Hyundai', 'Tucson'): '투싼',
    ('Hyundai', 'Kona'): '코나', ('Hyundai', 'Porter II'): '포터', ('Hyundai', 'Staria'): '스타리아',
    ('Hyundai', 'Ioniq 5'): '아이오닉 5', ('Genesis', 'G80'): '제네시스는 G80', ('Genesis', 'GV70'): 'GV70',
    ('Genesis', 'GV80'): 'GV80', ('Kia', 'Sorento'): '쏘렌토', ('Kia', 'Ray'): '레이', ('Kia', 'K5'): 'K5',
    ('Kia', 'K8'): 'K8', ('Kia', 'Seltos'): '셀토스', ('Kia', 'Carnival'): '카니발', ('Kia', 'Sportage'): '스포티지',
    ('Kia', 'Picanto'): '모닝', ('Kia', 'Bongo 3'): '봉고Ⅲ', ('Kia', 'PV5'): 'PV5', ('Kia', 'Niro'): '니로',
}
PRESS_NOTE = {('Hyundai', 'Elantra'): 'press figure includes Avante N (our Elantra N is separate)',
              ('Kia', 'Niro'): 'press figure includes Niro Plus (not in our series)'}
DN_EXTRA = {  # reviewed new Danawa line names (added after the audit; seen in the 2026-09 format check)
    ('Hyundai', 'Elantra'): ['디 올 뉴 아반떼'],   # CN8 = IR 'Avante (CN8)', already in our Elantra definition
    ('Hyundai', 'Ioniq 6'): ['아이오닉 6 N'],     # IR 'IONIQ 6 N (CE N)' is part of our Ioniq 6
}
DN_NOTE = {
    ('Kia', 'Bongo 3'): "Danawa line dropped the special-body (특장차) trim from 2026-05 and differed from the Kia release "
                        "before; raw follows the Kia release 봉고Ⅲ since the 2026-10-09 fix -> Danawa is fallback only",
    ('Kia', 'Niro'): "니로 플러스 (taxi/PBV EV) excluded; '니로' line already contains the EV trim (KR-03: do not add the EV trim again)",
    ('Hyundai', 'Kona'): "'코나 N' excluded (no Kona N series)", ('Hyundai', 'Casper'): "'캐스퍼 일렉트릭' = our Inster",
    ('Hyundai', 'Santa Fe'): "'맥스크루즈' excluded", ('KG Mobility', 'Tivoli'): "'티볼리 에어' excluded (KR-07)",
    ('KG Mobility', 'Musso'): "'무쏘 스포츠'/'무쏘 칸' (2025) are not in our series; listed names keep 2026 definition",
    ('Hyundai', 'Staria'): "'그랜드 스타렉스'/'스타렉스' excluded", ('Kia', 'K8'): "'K7' lines excluded",
    ('Hyundai', 'Sonata'): 'includes the taxi (DN8 LPG); IR rows exclude it from 2024 -> Danawa is primary',
    ('Genesis', 'GV60'): 'IR has no MAGMA split -> Danawa primary for GV60 and GV60 MAGMA',
}


def main():
    dnc, imp, dn = load_defs()
    raw = pd.read_csv(os.path.join(ROOT, 'raw', 'data.csv'))
    k = raw[(raw.Country == 'South Korea') & (raw.Month <= '2026-07')]
    last12 = k[k.Month >= '2025-08'].groupby(['Brand', 'Model']).Sales.sum()
    series = k.groupby(['Brand', 'Model']).Sales.sum().index
    bid = dn.drop_duplicates(['nat', 'name']).set_index(['nat', 'name']).brand_id.to_dict()
    # Brand of import lines: audit IMP keys are our (Brand, Model); brand_id from the line names
    rows = []

    def add(b, m, sid, key, rule, prio, conf, note=''):
        rows.append(dict(Country='South Korea', Brand=b, Model=m, source_id=sid, source_key=key, rule=rule,
                         confidence=conf, note=note, priority=prio))

    for b, m in series:
        key = (b, m)
        if key in HMC_DEF:
            prio = 9 if key in [('Hyundai', 'Sonata'), ('Genesis', 'GV60')] else 1
            add(b, m, 'KR_HMC_IR', '|'.join(HMC_DEF[key]), 'sum_rows', prio, 'high',
                HMC_EXCLUDED_NOTE.get(key, '') + ('; check only (definition differs)' if prio == 9 else ''))
        if b == 'KG Mobility' and m in KGM:
            add(b, m, 'KR_KGM_IR', '|'.join(KGM[m]), 'sum_rows', 1, 'high', KGM_NOTE.get(m, ''))
        if key in PRESS:
            prio = 1 if key == ('Kia', 'Bongo 3') else 9
            add(b, m, 'KR_HMG_PRESS', ('Hyundai' if b in ('Hyundai', 'Genesis') else 'Kia') + ':' + PRESS[key], 'named_in_release',
                prio, 'high' if prio == 1 else 'medium',
                PRESS_NOTE.get(key, 'release names only the top models of each segment; check only' if prio == 9 else
                               'Kia release 봉고Ⅲ = definition used in raw since 2026-10-09'))
        if key in dnc:
            names = list(dnc[key])
            if key == ('Kia', 'Niro'):
                names = ['니로', '니로 EV', '니로 플러그인 하이브리드']
            if key == ('KG Mobility', 'Musso'):
                names = ['무쏘', '무쏘 EV']
            names += [n for n in DN_EXTRA.get(key, []) if n not in names]
            hb = bid.get(('domestic', names[0]), '?')
            parts = [f"{bid.get(('domestic', n), hb)}:{n}" for n in names]
            has_primary = key in HMC_DEF and key not in [('Hyundai', 'Sonata'), ('Genesis', 'GV60')] or \
                (b == 'KG Mobility' and m in KGM) or key == ('Kia', 'Bongo 3')
            add(b, m, 'KR_DANAWA_DOMESTIC', '|'.join(parts), 'sum_lines_absent0', 2 if has_primary else 1,
                'low' if key == ('Kia', 'Bongo 3') else 'medium', DN_NOTE.get(key, ''))
        elif key in imp:
            defn = imp[key]['defn']
            parts = [f"{bid.get(('export', n), '?')}:{n}" for n in defn]
            other = sorted(set(imp[key]['lines']) - set(defn))
            add(b, m, 'KR_DANAWA_IMPORT', '|'.join(parts), 'sum_lines_absent0', 1, 'medium',
                ('lines not counted: ' + ', '.join(other)) if other else '')
        elif key == ('Ferrari', '12Cilindri'):
            pass
    mp = pd.DataFrame(rows)
    mp['sales_last12'] = [last12.get((b, m), 0.0) for b, m in zip(mp.Brand, mp.Model)]
    got = set(zip(mp.Brand, mp.Model))
    missing = [s for s in series if s not in got]
    for b, m in missing:
        add(b, m, 'NONE', '', 'no_source', 0, 'low', 'no source line found in the audit (list only)')
    mp = pd.DataFrame(rows)
    mp['sales_last12'] = [last12.get((b, m), 0.0) for b, m in zip(mp.Brand, mp.Model)]
    mp = mp.sort_values(['sales_last12', 'Brand', 'Model', 'priority'], ascending=[False, True, True, True])
    mp.to_csv(os.path.join(HERE, 'mapping.csv'), index=False, encoding='utf-8-sig')
    print(len(mp), 'rows;', mp[['Brand', 'Model']].drop_duplicates().shape[0], 'series; missing:', missing)
    print(mp.groupby(['source_id', 'priority']).size())
    bad = mp[mp.source_key.str.contains(r'\?:', regex=True)]
    if len(bad):
        print('unknown brand ids:\n', bad[['Brand', 'Model', 'source_key']].to_string())


if __name__ == '__main__':
    main()
