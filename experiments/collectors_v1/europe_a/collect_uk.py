"""United Kingdom (DfT VEH0160 quarterly by model, split to months with DfT VEH1153 monthly) -> cars by make/model/month.

Usage:
    python collect_uk.py --origin 2026-07 [--months 4] [--out out/uk_model_month.csv] [--refresh]

Sources (both GOV.UK, Open Government Licence; robots.txt allows /media/):
  VEH0160  df_VEH0160_UK.csv  - cars registered for the first time, by Make / GenModel / Model / Fuel,
                                 one column per quarter (2014 Q3 ..). Released ~80 days after quarter end.
  VEH1153  veh1153.ods        - registered for the first time by body type & fuel, *monthly* (UK from 2014-07),
                                 sheet VEH1153a_RoadUsing, values in thousands with 3 decimals (= units).
  Links are read from the two GOV.UK index pages at run time (asset URLs change every release).

Quarter -> month split rule (split_rules.csv, rule UK_Q2M_FUEL):
  units[model, fuel, month] = VEH0160[model, fuel, quarter]
                              * VEH1153[Cars, UK, Keepership=Total, fuel, month]
                              / sum_{months of quarter} VEH1153[Cars, UK, Total, fuel, month]
  i.e. every model of a fuel type gets that fuel type's UK monthly profile inside the quarter
  (captures the March/September plate-change peaks and BEV vs petrol timing; does NOT capture
  model-specific launches or stock pushes). Fallback when the fuel's quarter total is 0: equal thirds.
  Check: VEH0160 Cars 2026 Q2 = 538,867 = VEH1153 Cars Apr+May+Jun 2026 (154,481+164,987+219,399).

Scope: "registered for the first time" = new cars + a small number of used imports (DfT does not split).
Months whose quarter is not yet published are not output (UK 2026-07 is in Q3 -> not available
until ~2026-12).
Output: country, month, make, model, units, gen_model, dft_model, fuel, quarter, quarter_units, month_share
  make = DfT Make, model = DfT GenModel (with make prefix removed), dft_model = DfT Model (variant text).
"""
import argparse
import csv
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache', 'uk')
UA = 'daizen-research/1.0 (monthly statistics collector)'
PAGE_0160 = 'https://www.gov.uk/government/statistical-data-sets/vehicle-licensing-statistics-data-files'
PAGE_1153 = 'https://www.gov.uk/government/statistical-data-sets/vehicle-licensing-statistics-data-tables'

FUEL_TO_1153 = {
    'Petrol': 'Petrol', 'Diesel': 'Diesel',
    'Hybrid electric (petrol)': 'Hybrid Electric - Petrol',
    'Hybrid electric (diesel)': 'Hybrid Electric - Diesel',
    'Plug-in hybrid electric (petrol)': 'Plug-in Hybrid Electric - Petrol',
    'Plug-in hybrid electric (diesel)': 'Plug-in Hybrid Electric - Diesel',
    'Battery electric': 'Battery Electric', 'Range extended electric': 'Range Extended Electric',
    'Fuel cell electric': 'Fuel Cell Electric', 'Gas': 'Gas', 'Other fuel types': 'Others',
}
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December']


def _get(url, last=[0.0]):
    wait = 2.0 - (time.time() - last[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            sys.exit('HTTP 429 from GOV.UK -> stop')
        raise
    finally:
        last[0] = time.time()


def ensure_files(refresh):
    os.makedirs(CACHE, exist_ok=True)
    f0160 = os.path.join(CACHE, 'dft_df_VEH0160_UK.csv')
    f1153 = os.path.join(CACHE, 'veh1153.ods')
    if refresh or not os.path.exists(f0160):
        page = _get(PAGE_0160).decode('utf-8', 'replace')
        url = re.search(r'href="(https://assets\.publishing\.service\.gov\.uk/[^"]*df_VEH0160_UK\.csv)"', page).group(1)
        open(f0160, 'wb').write(_get(url))
    if refresh or not os.path.exists(f1153):
        page = _get(PAGE_1153).decode('utf-8', 'replace')
        url = re.search(r'href="(https://assets\.publishing\.service\.gov\.uk/[^"]*veh1153\.ods)"', page).group(1)
        open(f1153, 'wb').write(_get(url))
        f = os.path.join(CACHE, 'veh1153a_raw.csv')
        if os.path.exists(f):
            os.remove(f)
    return f0160, f1153


def monthly_fuel_profile(f1153):
    """{(YYYY-MM, VEH1153 fuel column): units} for UK cars, all keepers. Converts the slow ODS once."""
    import pandas as pd
    raw = os.path.join(CACHE, 'veh1153a_raw.csv')
    if not os.path.exists(raw):
        v = pd.read_excel(f1153, sheet_name='VEH1153a_RoadUsing', engine='odf', header=None)
        v.to_csv(raw, index=False, header=False)
    v = pd.read_csv(raw, header=None, low_memory=False)
    hdr_row = v.index[v[0] == 'Geography'][0]
    v.columns = v.iloc[hdr_row].tolist()
    v = v.iloc[hdr_row + 1:]
    v = v[(v['Geography'] == 'United Kingdom') & (v['Date Interval'] == 'Monthly')
          & (v['BodyType'] == 'Cars') & (v['Keepership'] == 'Total')]
    out = {}
    for _, r in v.iterrows():
        mon, yr = r['Date'].split()
        ym = f'{int(yr)}-{MONTHS.index(mon) + 1:02d}'
        for col in FUEL_TO_1153.values():
            val = str(r[col]).strip()
            out[(ym, col)] = round(float(val) * 1000) if val not in ('', 'nan', '[low]', '[x]') else 0
    return out


def quarter_of(ym):
    y, m = map(int, ym.split('-'))
    return f'{y} Q{(m - 1) // 3 + 1}', [f'{y}-{(q - 1) // 3 * 3 + i + 1:02d}' for q in [m] for i in range(3)]


def month_range(origin, n):
    y, m = map(int, origin.split('-'))
    out = []
    for _ in range(n):
        out.append(f'{y}-{m:02d}')
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--origin', required=True)
    ap.add_argument('--months', type=int, default=1)
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'uk_model_month.csv'))
    a = ap.parse_args()
    f0160, f1153 = ensure_files(a.refresh)
    prof = monthly_fuel_profile(f1153)
    import pandas as pd
    u = pd.read_csv(f0160, low_memory=False)
    u = u[u['BodyType'] == 'Cars']
    rows = []
    for ym in month_range(a.origin, a.months):
        q, qmonths = quarter_of(ym)
        if q not in u.columns:
            print(f'{ym}: quarter {q} not published in VEH0160 -> skipped', file=sys.stderr)
            continue
        vals = pd.to_numeric(u[q], errors='coerce').fillna(0)
        n_rows = 0
        for (_, r), qv in zip(u.iterrows(), vals):
            if qv <= 0:
                continue
            col = FUEL_TO_1153[r['Fuel']]
            qsum = sum(prof.get((m, col), 0) for m in qmonths)
            share = prof.get((ym, col), 0) / qsum if qsum > 0 else 1 / 3
            make = str(r['Make']).strip()
            gen = str(r['GenModel']).strip()
            model = gen[len(make):].strip() if gen.upper().startswith(make.upper() + ' ') else gen
            rows.append(('United Kingdom', ym, make, model, round(qv * share, 2), gen,
                         str(r['Model']).strip(), r['Fuel'], q, int(qv), round(share, 5)))
            n_rows += 1
        print(f'{ym}: {n_rows} rows, {sum(x[4] for x in rows if x[1] == ym):.0f} cars (from {q})', file=sys.stderr)
    rows.sort(key=lambda r: (r[1], -r[4], r[2], r[3]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['country', 'month', 'make', 'model', 'units', 'gen_model', 'dft_model', 'fuel',
                    'quarter', 'quarter_units', 'month_share'])
        w.writerows(rows)
    print('wrote', a.out, len(rows), 'rows', file=sys.stderr)


if __name__ == '__main__':
    main()
