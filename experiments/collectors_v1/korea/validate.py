"""Compare the collector with raw/data.csv (read only) for months <= 2026-07, offline (cache only).

python validate.py                      # 2026-05..2026-07 (SPEC2 test months) + 2025-08..2026-04 (extra, cache only)
Writes validation.csv (chosen value per series x month, plus each source's own value) and prints match rates.
EXACT = same number; MINOR = |diff| <= max(2, 1%); DIFF otherwise.
"""
import os
import sys

import pandas as pd

import collect_korea as C

TEST = ['2026-05', '2026-06', '2026-07']
EXTRA = [str(p) for p in pd.period_range('2025-08', '2026-04', freq='M')]


def status(o, c):
    if pd.isna(c):
        return 'NOT_COLLECTED'
    if pd.isna(o):
        return 'NOT_IN_RAW'
    d = abs(o - c)
    return 'EXACT' if d < 0.5 else ('MINOR' if d <= max(2.0, 0.01 * abs(o)) else 'DIFF')


def main():
    raw = pd.read_csv(C.RAW)
    raw = raw[(raw.Country == 'South Korea') & (raw.Month <= C.LAST_ALLOWED)]
    ours = raw.set_index(['Brand', 'Model', 'Month']).Sales
    out = []
    for month in TEST + EXTRA:
        std, cand, unm, checks = C.collect(month, offline=True)
        piv = cand.pivot_table(index=['Brand', 'Model'], columns='source_id', values='value', aggfunc='first')
        s = std.set_index(['Brand', 'Model'])
        keys = set(piv.index) | {(b, m) for b, m, p in ours.index if p == month}
        for b, m in keys:
            o = ours.get((b, m, month))
            c = s.Sales.get((b, m)) if (b, m) in s.index else None
            rec = dict(Country='South Korea', Brand=b, Model=m, Month=month, set='test' if month in TEST else 'extra',
                       ours=o, collected=c, diff=(c - o) if c is not None and o is not None and pd.notna(o) else None,
                       status=status(o, c),
                       source_id=s.source_id.get((b, m)) if (b, m) in s.index else '',
                       confidence=s.confidence.get((b, m)) if (b, m) in s.index else '',
                       note=s.note.get((b, m)) if (b, m) in s.index else '')
            for sid in piv.columns:
                v = piv.loc[(b, m), sid] if (b, m) in piv.index else None
                rec['v_' + sid] = v
                rec['st_' + sid] = status(o, v) if v is not None and pd.notna(v) else ''
            out.append(rec)
        print(month, 'done', file=sys.stderr)
    v = pd.DataFrame(out).sort_values(['set', 'Month', 'Brand', 'Model'], ascending=[False, True, True, True])
    v.to_csv(os.path.join(C.HERE, 'validation.csv'), index=False, encoding='utf-8-sig')
    for st in ('test', 'extra'):
        x = v[(v.set == st) & v.ours.notna()]
        w = x.ours.abs()
        print(f'\n== {st} months ({x.Month.min()}..{x.Month.max()}): {len(x)} series-months, raw volume {w.sum():,.0f}')
        print(f"chosen value: EXACT {(x.status == 'EXACT').mean():.1%} of rows / {w[x.status == 'EXACT'].sum() / w.sum():.2%} of volume;"
              f" EXACT+MINOR {w[x.status.isin(['EXACT', 'MINOR'])].sum() / w.sum():.2%} of volume; not collected {(x.status == 'NOT_COLLECTED').sum()}")
        for sid, g in x.groupby('source_id'):
            gw = g.ours.abs()
            print(f'  chosen {sid:20s} rows {len(g):4d} vol {gw.sum():9,.0f}  exact {(g.status == "EXACT").mean():6.1%} rows,'
                  f' {gw[g.status == "EXACT"].sum() / max(gw.sum(), 1):6.2%} vol; DIFF rows {(g.status == "DIFF").sum()}')
        for c in [c for c in v.columns if c.startswith('st_')]:
            g = x[x[c] != '']
            if not len(g):
                continue
            gw = g.ours.abs()
            print(f'  own value {c[3:]:20s} rows {len(g):4d}  exact {(g[c] == "EXACT").mean():6.1%} rows,'
                  f' {gw[g[c] == "EXACT"].sum() / max(gw.sum(), 1):6.2%} vol; exact+minor {gw[g[c].isin(["EXACT", "MINOR"])].sum() / max(gw.sum(), 1):6.2%} vol')
        d = x[x.status.isin(['DIFF', 'MINOR', 'NOT_COLLECTED'])]
        if len(d):
            print(d[['Month', 'Brand', 'Model', 'ours', 'collected', 'source_id', 'status']].to_string(index=False))


if __name__ == '__main__':
    main()
