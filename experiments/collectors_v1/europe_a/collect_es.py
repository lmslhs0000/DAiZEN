"""Spain (DGT microdatos, matriculaciones mensuales) -> new passenger cars by make/model/month.

Usage:
    python collect_es.py --origin 2026-07 [--months 4] [--out out/es_model_month.csv]

- Downloads export_mensual_mat_YYYYMM.zip for each month in (origin-months+1 .. origin)
  into cache/es/ (skips files already cached). One request per file, >= 2 s between requests.
- Fixed-width latin-1 record per vehicle (layout = DGT "diseno de registro MATRABA";
  field offsets verified on the 2026-08 file, see README).
- New passenger car filter: IND_NUEVO_USADO == 'N' and CATEGORIA_HOMOLOGACION in {M1, M1G}.
  (Used imports 'U' and vans N1 are excluded.)
- Output columns: country, month, make, model, units   (model = MODELO_ITV text, trimmed).
  Month = month of FEC_MATRICULA (registration date). The file for month M only holds
  registrations dated in M (checked: 2026-08 file).
"""
import argparse
import io
import os
import sys
import time
import urllib.request
import zipfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache', 'es')
URL = 'https://www.dgt.es/microdatos/salida/{y}/{m}/vehiculos/matriculaciones/export_mensual_mat_{y}{m:02d}.zip'
UA = 'daizen-research/1.0 (monthly statistics collector)'

# (name, length) in order; only the first part of the record is needed.
LAYOUT = [
    ('FEC_MATRICULA', 8), ('COD_CLASE_MAT', 1), ('FEC_TRAMITACION', 8), ('MARCA_ITV', 30),
    ('MODELO_ITV', 22), ('COD_PROCEDENCIA_ITV', 1), ('BASTIDOR_ITV', 21), ('COD_TIPO', 2),
    ('COD_PROPULSION_ITV', 1), ('CILINDRADA_ITV', 5), ('POTENCIA_ITV', 6), ('TARA', 6),
    ('PESO_MAX', 6), ('NUM_PLAZAS', 3), ('IND_PRECINTO', 2), ('IND_EMBARGO', 2),
    ('NUM_TRANSMISIONES', 2), ('NUM_TITULARES', 2), ('LOCALIDAD_VEHICULO', 24),
    ('COD_PROVINCIA_VEH', 2), ('COD_PROVINCIA_MAT', 2), ('CLAVE_TRAMITE', 1), ('FEC_TRAMITE', 8),
    ('CODIGO_POSTAL', 5), ('FEC_PRIM_MATRICULACION', 8), ('IND_NUEVO_USADO', 1),
    ('PERSONA_FISICA_JURIDICA', 1), ('CODIGO_ITV', 9), ('SERVICIO', 3),
    ('COD_MUNICIPIO_INE_VEH', 5), ('MUNICIPIO', 30), ('KW_ITV', 7), ('NUM_PLAZAS_MAX', 3),
    ('CO2_ITV', 5), ('RENTING', 1), ('COD_TUTELA', 1), ('COD_POSESION', 1), ('IND_BAJA_DEF', 1),
    ('IND_BAJA_TEMP', 1), ('IND_SUSTRACCION', 1), ('BAJA_TELEMATICA', 11), ('TIPO_ITV', 25),
    ('VARIANTE_ITV', 25), ('VERSION_ITV', 35), ('FABRICANTE_ITV', 70), ('MASA_ORDEN_MARCHA', 6),
    ('MASA_MAXIMA', 6), ('CATEGORIA_HOMOLOGACION', 4),
]
OFFSETS = {}
_p = 0
for _n, _l in LAYOUT:
    OFFSETS[_n] = (_p, _p + _l)
    _p += _l

PASSENGER = {'M1', 'M1G'}


def field(line, name):
    a, b = OFFSETS[name]
    return line[a:b].strip()


def month_range(origin, n):
    y, m = map(int, origin.split('-'))
    out = []
    for _ in range(n):
        out.append((y, m))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return sorted(out)


def fetch(y, m, last=[0.0]):
    path = os.path.join(CACHE, f'dgt_export_mensual_mat_{y}{m:02d}.zip')
    if os.path.exists(path):
        return path
    wait = 2.0 - (time.time() - last[0])
    if wait > 0:
        time.sleep(wait)
    url = URL.format(y=y, m=m)
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        last[0] = time.time()
        if e.code == 429:
            sys.exit('HTTP 429 from dgt.es -> stop (site etiquette)')
        print(f'{y}-{m:02d}: HTTP {e.code} (not published yet?)', file=sys.stderr)
        return None
    last[0] = time.time()
    os.makedirs(CACHE, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    return path


def parse(path):
    """Return Counter[(month, make, model)] of new passenger cars and a diagnostics dict."""
    counts = Counter()
    diag = Counter()
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as raw:
            f = io.TextIOWrapper(raw, encoding='latin-1', newline='')
            f.readline()  # title line
            for line in f:
                if len(line) < OFFSETS['CATEGORIA_HOMOLOGACION'][1]:
                    diag['short_line'] += 1
                    continue
                diag['records'] += 1
                cat = field(line, 'CATEGORIA_HOMOLOGACION')
                nu = field(line, 'IND_NUEVO_USADO')
                if cat not in PASSENGER:
                    continue
                diag[f'M1_{nu}'] += 1
                if nu != 'N':
                    continue
                d = field(line, 'FEC_MATRICULA')  # ddmmyyyy
                month = f'{d[4:8]}-{d[2:4]}'
                counts[(month, field(line, 'MARCA_ITV'), field(line, 'MODELO_ITV'))] += 1
    return counts, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--origin', required=True, help='last month to collect, YYYY-MM')
    ap.add_argument('--months', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'es_model_month.csv'))
    a = ap.parse_args()
    rows = []
    for y, m in month_range(a.origin, a.months):
        path = fetch(y, m)
        if path is None:
            continue
        counts, diag = parse(path)
        tag = f'{y}-{m:02d}'
        print(tag, dict(diag), 'new M1 kept:', sum(counts.values()), file=sys.stderr)
        for (month, make, model), n in counts.items():
            if month != tag:
                print(f'  note: record dated {month} in file {tag} ({n})', file=sys.stderr)
            rows.append(('Spain', month, make, model, n))
    rows.sort(key=lambda r: (r[1], -r[4], r[2], r[3]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    import csv
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['country', 'month', 'make', 'model', 'units'])
        w.writerows(rows)
    print('wrote', a.out, len(rows), 'rows', file=sys.stderr)


if __name__ == '__main__':
    main()
