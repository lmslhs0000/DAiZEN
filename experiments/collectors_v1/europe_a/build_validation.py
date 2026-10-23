"""validation.csv: per our Europe series (or series group) x month: our Europe value vs DE + ES + NL + UK.

Usage: python build_validation.py [--months 2026-04,2026-05,2026-06,2026-07]
Needs: build_mapping.py outputs (mapping_europe_a.csv, out/de_links.csv, out/de_kba_fallback.csv)
       and collector outputs in out/.  Only months <= 2026-07 are allowed (SPEC).

Groups: when one KBA line behind a Germany series covers several Europe series (e.g. KBA YARIS =
Yaris + Yaris Cross), those Europe series are validated together as one group (union-find).
share = (DE+ES+NL+UK) / our Europe. These four countries were 48.3 % of Europe (EU+EFTA+UK) in 2025
(ACEA: DE 21.5, UK 15.2, ES 8.7, NL 2.9), so a typical series should show share ~0.3-0.7;
flags: OVER (>1.05: impossible unless our Europe value is too low or a mapping is too wide),
HIGH (0.85-1.05), LOW (<0.10, Europe value mostly from countries we do not collect, or mapping gap).
UK months are quarter values split by the UK fuel-type monthly profile (estimate, flag uk_split=1);
for 2026-07 UK is not yet published (UK blank, share computed on DE+ES+NL only, flag uk_missing=1).
A quarter row (month='2026-Q2') is added per group, where UK is the exact DfT quarter value.
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
AUDIT = os.path.join(ROOT, 'experiments', 'data_audit_v1')


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', default='2026-04,2026-05,2026-06,2026-07')
    ap.add_argument('--raw', default=os.path.join(ROOT, 'raw', 'data.csv'))
    a = ap.parse_args()
    months = a.months.split(',')
    assert max(months) <= '2026-07', 'validation only on months <= 2026-07'

    raw = pd.read_csv(a.raw)
    eu = raw[(raw['Country'] == 'Europe') & raw['Month'].isin(months)]
    series = sorted(set(zip(raw.loc[raw['Country'] == 'Europe', 'Brand'], raw.loc[raw['Country'] == 'Europe', 'Model'])))
    mp = pd.read_csv(os.path.join(HERE, 'mapping_europe_a.csv'), keep_default_na=False)
    de = pd.read_csv(os.path.join(HERE, 'out', 'de_links.csv'), keep_default_na=False)
    kf = pd.read_csv(os.path.join(HERE, 'out', 'de_kba_fallback.csv'), keep_default_na=False)

    uf = UF()
    for s in series:
        uf.find(s)
    de_ok = de[de['Brand'] != '']
    for _, g in de_ok.groupby(['de_brand', 'de_model']):
        mem = list(zip(g['Brand'], g['Model']))
        for x in mem[1:]:
            uf.union(mem[0], x)
    gid = {s: uf.find(s) for s in series}
    members = {}
    for s, r in gid.items():
        members.setdefault(r, []).append(s)

    vals = {}  # (group_root, month, col) -> value

    def add(s, month, col, v):
        k = (gid[s], month, col)
        vals[k] = vals.get(k, 0.0) + float(v)

    for r in eu.itertuples(index=False):
        add((r.Brand, r.Model), r.Month, 'our_europe', r.Sales)
    # DE from our Germany series (each Germany series counted once, on its group)
    ger = raw[(raw['Country'] == 'Germany') & raw['Month'].isin(months)]
    first_target = {}
    for r in de_ok.itertuples(index=False):
        first_target.setdefault((r.de_brand, r.de_model), (r.Brand, r.Model))
    de_flag = {}
    for r in de_ok.itertuples(index=False):
        if r.confidence == 'low':
            de_flag[gid[(r.Brand, r.Model)]] = r.note
    for r in ger.itertuples(index=False):
        t = first_target.get((r.Brand, r.Model))
        if t:
            add(t, r.Month, 'DE', r.Sales)
    kl = pd.read_csv(os.path.join(AUDIT, 'kba_fz10_long_2016_2026-07.csv'))
    kl = kl[kl['period'].isin(months) & (kl['row_type'] == 'model')]
    for r in kf.itertuples(index=False):
        x = kl[(kl['kba_brand'] == r.kba_brand) & (kl['kba_model'] == r.kba_model)]
        for y in x.itertuples(index=False):
            add((r.Brand, r.Model), y.period, 'DE', y.sales)
    # ES / NL / UK
    srcs = {'ES': ('es_model_month.csv', 'ES_DGT_MATRABA'), 'NL': ('nl_model_month.csv', 'NL_RDW'),
            'UK': ('uk_model_month.csv', 'UK_DFT_VEH0160')}
    for col, (fn, sid) in srcs.items():
        d = pd.read_csv(os.path.join(HERE, 'out', fn), keep_default_na=False)
        d = d[d['month'].isin(months)]
        m = mp[(mp['source_id'] == sid) & (mp['Model'] != '')][['source_key', 'Brand', 'Model']]
        if col == 'UK':
            d['source_key'] = d['make'] + ' | ' + d['model'] + ' | ' + d['dft_model'] + ' | ' + d['fuel']
        else:
            d['source_key'] = d['make'] + ' | ' + d['model']
        j = d.merge(m, on='source_key', how='inner')
        for (b, mo, mon), v in j.groupby(['Brand', 'Model', 'month'])['units'].sum().items():
            add((b, mo), mon, col, v)

    rows = []
    for root, mem in members.items():
        mem = sorted(mem)
        label_b = mem[0][0] if len({b for b, _ in mem}) == 1 else ' / '.join(sorted({b for b, _ in mem}))
        label_m = ' + '.join(m for _, m in mem)
        q = {c: 0.0 for c in ['our_europe', 'DE', 'ES', 'NL', 'UK']}
        for mon in months + ['2026-Q2']:
            if mon == '2026-Q2':
                rec = q
            else:
                rec = {c: vals.get((root, mon, c), 0.0) for c in q}
                if mon in ('2026-04', '2026-05', '2026-06'):
                    for c in q:
                        q[c] += rec[c]
            eu_v = rec['our_europe']
            uk_missing = mon > '2026-06' and mon != '2026-Q2'
            s = rec['DE'] + rec['ES'] + rec['NL'] + (0 if uk_missing else rec['UK'])
            share = s / eu_v if eu_v > 0 else None
            flag = ''
            if eu_v <= 0 and s > 0:
                flag = 'NO_EUROPE_VALUE'
            elif share is not None:
                flag = 'OVER' if share > 1.05 else 'HIGH' if share > 0.85 else 'LOW' if share < 0.10 else 'ok'
            rows.append(dict(Brand=label_b, Model=label_m, n_series=len(mem), month=mon,
                             our_europe=round(eu_v, 1), DE=round(rec['DE'], 1), ES=round(rec['ES'], 1),
                             NL=round(rec['NL'], 1), UK='' if uk_missing else round(rec['UK'], 1),
                             sum=round(s, 1), share=round(share, 4) if share is not None else '',
                             flag=flag, uk_split=int(mon in ('2026-04', '2026-05', '2026-06')),
                             uk_missing=int(uk_missing), note=de_flag.get(root, '')))
    v = pd.DataFrame(rows)
    v = v[(v['our_europe'] > 0) | (v['sum'] > 0)]
    order = v[v['month'] == '2026-Q2'].sort_values('our_europe', ascending=False)['Model'].tolist()
    v['o'] = v['Model'].map({m: i for i, m in enumerate(order)})
    v = v.sort_values(['o', 'month']).drop(columns='o')
    v.to_csv(os.path.join(HERE, 'validation.csv'), index=False, encoding='utf-8')

    # summary
    qv = v[v['month'] == '2026-Q2'].copy()
    qv = qv[qv['our_europe'] > 0]
    tot = qv[['our_europe', 'DE', 'ES', 'NL', 'UK', 'sum']].apply(pd.to_numeric).sum()
    print('Q2-2026 all groups:', tot.round(0).to_dict(), 'share', round(tot['sum'] / tot['our_europe'], 4))
    print('groups', len(qv), 'flags', qv['flag'].value_counts().to_dict())
    qs = qv.assign(sh=pd.to_numeric(qv['share'])).sort_values('sh')
    cw = qs['our_europe'].cumsum() / qs['our_europe'].sum()
    print('volume-weighted median share', round(float(qs.loc[cw >= 0.5, 'sh'].iloc[0]), 3))
    top = qv.head(100).assign(sh=pd.to_numeric(qv.head(100)['share']))
    print('top-100 groups by Europe volume: share p10/p50/p90',
          top['sh'].quantile([.1, .5, .9]).round(3).tolist(), 'flags', top['flag'].value_counts().to_dict())
    print(qv.head(40)[['Brand', 'Model', 'our_europe', 'DE', 'ES', 'NL', 'UK', 'share', 'flag']].to_string(index=False))


if __name__ == '__main__':
    main()
