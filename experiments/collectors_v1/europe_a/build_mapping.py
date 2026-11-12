"""Build mapping_europe_a.csv: every ES/NL/UK source key -> our Europe series, plus DE (Germany) links.

Usage: python build_mapping.py [--raw ../../../raw/data.csv]
Inputs : out/es_model_month.csv, out/nl_model_month.csv, out/uk_model_month.csv (collector outputs),
         raw/data.csv (our series), experiments/data_audit_v1/germany_kba_mapping.csv,
         experiments/data_audit_v1/kba_fz10_long_2016_2026-07.csv
Output : mapping_europe_a.csv (SPEC columns + units_in_window), out/de_links.csv (used by build_validation.py)
"""
import argparse
import os
import re

import pandas as pd

from europe_a_mapping import Matcher, our_series, norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
AUDIT = os.path.join(ROOT, 'experiments', 'data_audit_v1')

# Germany series (our Germany market, = KBA FZ10 via germany_kba_mapping.csv) -> Europe series.
# Identity by name is automatic (brand aliases below); these are the exceptions where the KBA line
# behind our Germany series covers several (or differently named) Europe series.
DE_BRAND = {'Opel': 'Opel/Vauxhall', 'Mini': 'MINI', 'NIO': 'Nio'}
DE_GROUPS = {
    ('Toyota', 'Yaris'): ['Yaris', 'Yaris Cross'], ('Toyota', 'Aygo'): ['Aygo X'],
    ('Toyota', 'Corolla'): ['Corolla', 'Corolla Cross'], ('Toyota', 'C-HR'): ['C-HR', 'C-HR+'],
    ('Toyota', 'Proace'): ['Proace', 'Proace City', 'Proace Max'],
    ('Mini', 'Mini'): ['Cooper', 'Countryman', 'Aceman'],
    ('Citroen', 'C3'): ['C3', 'C3 Aircross'], ('Citroen', 'C4'): ['C4', 'C4 X'],
    ('Citroen', 'C5 / C5 Aircross / C5 X'): ['C5 Aircross', 'C5 X'],
    ('Volkswagen', 'ID.4 / ID.5'): ['ID.4', 'ID.5'],
    ('Suzuki', 'Vitara'): ['Vitara/Grand Vitara', 'e Vitara'],
    ('Renault', 'Scenic'): ['Scenic E-Tech'], ('Renault', 'Megane'): ['Megane E-Tech'],
    ('Renault', 'Twingo'): ['Twingo E-Tech'],
    ('Audi', 'Q3'): ['Q3', 'Q3 Sportback'], ('Audi', 'Q4'): ['Q4', 'Q4 Sportback'],
    ('Audi', 'Q5'): ['Q5', 'Q5 Sportback'], ('Audi', 'Q6'): ['Q6 e-tron', 'Q6 Sportback e-tron'],
    ('Audi', 'A6'): ['A6', 'A6 e-tron'], ('Audi', 'Q8 e-tron'): ['Q8'],
    ('Volvo', 'XC40'): ['XC40', 'EX40'], ('Volvo', 'S60 / V60'): ['S60/V60'], ('Volvo', 'S90 / V90'): ['S90/V90'],
    ('Mercedes-Benz', 'GLA'): ['GLA', 'EQA'], ('Mercedes-Benz', 'GLB'): ['GLB', 'EQB'],
    ('Mercedes-Benz', 'E-Class'): ['E-Class', 'EQE'], ('Mercedes-Benz', 'GLE'): ['GLE', 'EQE SUV'],
    ('Mercedes-Benz', 'S-Class'): ['S-Class', 'EQS'], ('Mercedes-Benz', 'GLS'): ['GLS', 'EQS SUV'],
    ('Mercedes-Benz', 'V-Class'): ['V-Class', 'EQV'], ('Mercedes-Benz', 'Citan'): ['Citan', 'EQT'],
    ('Mercedes-Benz', 'AMG GT'): ['AMG GT', 'AMG GT 4 Door'],
    ('BMW', '2 Series'): ['2 Series', '2 Series Active Tourer'], ('BMW', 'X2'): ['X2', 'iX2'],
    ('BMW', 'X5'): ['X5', 'iX5'],
    ('Seat', 'Leon'): [('Seat', 'Leon'), ('Cupra', 'Leon')], ('Seat', 'Ateca'): [('Cupra', 'Ateca')],
    ('Kia', 'Ceed'): ['XCeed'], ('Mazda', 'Mazda6'): ['Mazda6e'], ('Subaru', 'Outback'): ['Legacy/Outback'],
    ('Porsche', 'Boxster'): ['718'], ('Porsche', 'Cayman'): ['718'],
    ('MG', 'MGS5 EV'): ['S5'], ('MG', 'MGS6 EV'): ['S6'], ('MG', 'MGS9'): ['S9'], ('MG', 'MG4'): ['MG4', 'MG4 Urban'],
    ('Geely', 'Starray EM-i'): ['Starray'], ('Ford', 'Transit / Tourneo'): ['Transit'],
    ('DS', '8'): ['N 8'],  # DS 8 renamed DS N°8 (resolved by normalised name)
    ('Skoda', 'Enyaq'): ['Enyaq', 'Enyaq Coupe'],  # ES/NL/UK also register the coupe as ENYAQ
    ('BYD', 'Seal'): ['Seal', 'Seal 6 DM-i', 'Seal 5 DM-i'],  # KBA has no SEAL 6 / SEAL 5 line
    ('Xpeng', 'P7'): ['P7', 'P7+'],
}
DE_LOW = {('Kia', 'Ceed'): 'KBA CEED line = Ceed+ProCeed+XCeed; Europe has only XCeed -> DE overstates',
          ('Seat', 'Ateca'): 'KBA SEAT ATECA = Seat+Cupra Ateca; Europe has only Cupra Ateca -> DE overstates',
          ('Mercedes-Benz', 'GLS'): 'EQS SUV assumed inside KBA GLS line (unverified)'}


def de_links(series, raw_csv):
    d = pd.read_csv(raw_csv)
    g = d[d['Country'] == 'Germany'][['Brand', 'Model']].drop_duplicates()
    km = pd.read_csv(os.path.join(AUDIT, 'germany_kba_mapping.csv'))
    km = km.set_index(['Brand', 'Model'])
    rows = []
    for b, m in map(tuple, g.values):
        eb = DE_BRAND.get(b, b)
        targets = DE_GROUPS.get((b, m))
        if targets is None:
            targets = [m] if (eb, m) in series else []
        tl = [(t if isinstance(t, tuple) else (eb, t)) for t in targets]
        tl = [t if t in series else next((x for x in series if x[0] == t[0] and norm(x[1]) == norm(t[1])), None)
              for t in tl]
        tl = [t for t in tl if t is not None]
        key = km.loc[(b, m), 'group_kba_lines'] if (b, m) in km.index else ''
        if not tl:
            rows.append(dict(Country='Europe', Brand='', Model='', source_id='DE_KBA_FZ10',
                             source_key=f'Germany series {b} | {m} (KBA {key})', rule='no_europe_series',
                             confidence='', note='', de_brand=b, de_model=m))
            continue
        for tb, tm in tl:
            rule = 'de_same_name' if (b, m) not in DE_GROUPS else ('de_group' if len(tl) > 1 else 'de_rename')
            conf = 'low' if (b, m) in DE_LOW else ('high' if rule != 'de_group' else 'medium')
            rows.append(dict(Country='Europe', Brand=tb, Model=tm, source_id='DE_KBA_FZ10',
                             source_key=f'Germany series {b} | {m} (KBA {key})', rule=rule, confidence=conf,
                             note=DE_LOW.get((b, m), 'value = raw/data.csv Germany (= KBA FZ10, BMW BEV split)'
                                             + ('; one KBA line -> several Europe series: validate as group'
                                                if len(tl) > 1 else '')),
                             de_brand=b, de_model=m))
    return pd.DataFrame(rows)


def kba_fallback(series, matcher, de):
    """KBA lines not used by any of our Germany series -> Europe series that have no Germany link."""
    kl = pd.read_csv(os.path.join(AUDIT, 'kba_fz10_long_2016_2026-07.csv'))
    kl = kl[(kl['row_type'] == 'model') & (kl['period'] >= '2025-01')]
    km = pd.read_csv(os.path.join(AUDIT, 'germany_kba_mapping.csv'))
    used = set()
    for x in km['group_kba_lines'].dropna():
        for part in str(x).split(' + '):
            used.add(part.strip())
    linked = set(zip(de['Brand'], de['Model']))
    rows = []
    for (kb, kmod), n in kl.groupby(['kba_brand', 'kba_model'])['sales'].sum().items():
        if f'{kb}/{kmod}' in used or kmod == 'SONSTIGE' or n <= 0:
            continue
        mk = 'Cupra' if kb == 'SEAT' else kb
        b, m, rule, conf = matcher.match(mk, kmod)
        if m is None or (b, m) in linked:
            continue
        rows.append(dict(Country='Europe', Brand=b, Model=m, source_id='DE_KBA_FZ10_LONG',
                         source_key=f'{kb} | {kmod}', rule='kba_line_' + rule, confidence='medium',
                         note='KBA line not used by our Germany series; value from data_audit_v1/kba_fz10_long',
                         kba_brand=kb, kba_model=kmod))
    return pd.DataFrame(rows)


def map_country(matcher, path, country, source_id):
    d = pd.read_csv(path, keep_default_na=False)
    if country == 'United Kingdom':
        g = d.groupby(['make', 'model', 'dft_model', 'fuel'])['units'].sum().reset_index()
    else:
        g = d.groupby(['make', 'model'])['units'].sum().reset_index()
    out = []
    for r in g.itertuples(index=False):
        if country == 'United Kingdom':
            b1, m1, r1, c1 = matcher.match(r.make, r.dft_model)
            b2, m2, r2, c2 = matcher.match(r.make, r.model)
            if m1 is not None or r1.startswith('regex_none'):
                b, m, rule, conf = b1, m1, 'dft_model:' + r1, c1
            else:
                b, m, rule, conf = b2, m2, 'gen_model:' + r2, c2
            if b == 'Volvo' and norm(r.model) == 'XC 40' and r.fuel == 'Battery electric':
                m, rule, conf = 'EX40', 'uk_fuel:XC40 BEV -> EX40', 'medium'
            if b == 'Volvo' and norm(r.model) in ('C 40',) and r.fuel == 'Battery electric':
                m, rule, conf = 'C40', 'uk_fuel', 'high'
            key = f'{r.make} | {r.model} | {r.dft_model} | {r.fuel}'
        else:
            b, m, rule, conf = matcher.match(r.make, r.model)
            key = f'{r.make} | {r.model}'
        note = ''
        if b is None:
            note = 'make not in our Europe brand list'
        elif m is None and 'regex_none' in rule:
            note = 'known model without an Europe series in raw/data.csv'
        elif m is None:
            note = 'unmatched'
        out.append(dict(Country='Europe', Brand=b if m else (b or ''), Model=m or '', source_id=source_id,
                        source_key=key, rule=rule if m or 'regex_none' in rule else (rule or 'unmapped'),
                        confidence=conf if m else '', note=note, units_in_window=round(r.units, 1),
                        src_make=r.make, src_model=r.model,
                        src_detail=getattr(r, 'dft_model', ''), src_fuel=getattr(r, 'fuel', '')))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=os.path.join(ROOT, 'raw', 'data.csv'))
    a = ap.parse_args()
    series = set(our_series(a.raw))
    matcher = Matcher(sorted(series))
    parts = [
        map_country(matcher, os.path.join(HERE, 'out', 'es_model_month.csv'), 'Spain', 'ES_DGT_MATRABA'),
        map_country(matcher, os.path.join(HERE, 'out', 'nl_model_month.csv'), 'Netherlands', 'NL_RDW'),
        map_country(matcher, os.path.join(HERE, 'out', 'uk_model_month.csv'), 'United Kingdom', 'UK_DFT_VEH0160'),
    ]
    de = de_links(series, a.raw)
    kf = kba_fallback(series, matcher, de[de['Brand'] != ''])
    de.to_csv(os.path.join(HERE, 'out', 'de_links.csv'), index=False)
    kf.to_csv(os.path.join(HERE, 'out', 'de_kba_fallback.csv'), index=False)
    cols = ['Country', 'Brand', 'Model', 'source_id', 'source_key', 'rule', 'confidence', 'note', 'units_in_window']
    m = pd.concat(parts + [de, kf], ignore_index=True)
    m['units_in_window'] = m['units_in_window'].fillna('')
    m = m.sort_values(['source_id', 'Brand', 'Model', 'units_in_window'], ascending=[True, True, True, False])
    m[cols].to_csv(os.path.join(HERE, 'mapping_europe_a.csv'), index=False, encoding='utf-8')
    for p in parts:
        tot = p['units_in_window'].sum()
        mapped = p.loc[p['Model'] != '', 'units_in_window'].sum()
        known_none = p.loc[p['rule'].str.contains('regex_none'), 'units_in_window'].sum()
        nob = p.loc[p['note'] == 'make not in our Europe brand list', 'units_in_window'].sum()
        print(p['source_id'].iloc[0], f'units {tot:.0f}  mapped {mapped / tot:.1%}  known-no-series {known_none / tot:.1%}'
              f'  brand-not-ours {nob / tot:.1%}  unmatched {(tot - mapped - known_none - nob) / tot:.1%}')
    print('DE links', len(de), 'KBA fallback lines', len(kf))


if __name__ == '__main__':
    main()
