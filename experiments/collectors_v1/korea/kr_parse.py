"""Parsers for the South Korea sources (no network here).

danawa_page(path)    -> (lines DataFrame[nat, model_id, name, brand_id, sales], trims DataFrame, brand names {id: name})
hmc_ir(path, year)   -> DataFrame[month, segment, name, units] (Hyundai IR 'Unit Sales by Model', Domestic block),
                        only months whose Domestic Total is filled (later columns of a year-to-date file are dropped)
kgm_ir(path, year)   -> DataFrame[month, name, units] (KGM IR sheet 'YYYY', Domestic CBU block), same month rule
hmg_press(path)      -> dict(maker 'Hyundai'|'Kia', month, models {name: units}, domestic_total, genesis_total)
kaida_press(path)    -> (month, {brand: units}, total)
"""
import html
import re

import openpyxl
import pandas as pd

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _text(raw):
    b = re.sub(r'<script.*?</script>|<style.*?</style>', '', raw, flags=re.S)
    return html.unescape(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', b)))


def _num(s):
    s = re.sub(r'[^0-9\-]', '', str(s))
    return float(s) if s not in ('', '-') else None


# ------------------------------------------------------------------ Danawa
ROW = re.compile(r"<tr>\s*<td><input type='checkbox' name='compItemCk' value='record_(\d+)' title='([^']*)' brand='(\d+)'>"
                 r".*?<td class='num'>([\d,]+)", re.S)
TRIM = re.compile(r"<tr class='sub model_(\d+)' name='sub_model'>.*?<td class='title'><span>(.*?)</span></td>\s*"
                  r"<td class='num'>([\d,]+)</td>", re.S)
BRAND = re.compile(r"data-brand='(\d+)'><button type='button'>.*?<span class='name'>(.*?)</span>", re.S)


def danawa_page(path, nat):
    t = open(path, encoding='utf-8', errors='ignore').read()
    lines = [dict(nat=nat, model_id=m.group(1), name=html.unescape(m.group(2)).strip(), brand_id=m.group(3),
                  sales=_num(m.group(4))) for m in ROW.finditer(t)]
    trims = [dict(nat=nat, model_id=m.group(1), trim=html.unescape(m.group(2)).strip(), sales=_num(m.group(3)))
             for m in TRIM.finditer(t)]
    brands = {b: html.unescape(n).strip() for b, n in BRAND.findall(t)}
    if not lines:
        raise ValueError(f'no model rows parsed from {path} (page layout changed?)')
    return (pd.DataFrame(lines, columns=['nat', 'model_id', 'name', 'brand_id', 'sales']),
            pd.DataFrame(trims, columns=['nat', 'model_id', 'trim', 'sales']), brands)


def danawa_month_shown(path):
    """Month the page says it shows (selected year/month options), to catch silent redirects to another month."""
    t = open(path, encoding='utf-8', errors='ignore').read()
    blk = re.search(r"<select id='selMonth'\s*>(.*?)</select>(.*?)</select>", t, re.S)
    if not blk:
        return None
    y = re.search(r"value='(\d{4})' selected", blk.group(0))
    m = re.search(r"value='(\d{2})' selected", blk.group(0))
    return f'{y.group(1)}-{m.group(1)}' if y and m else None


# ------------------------------------------------------------------ Hyundai IR
def hmc_ir(path, year):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb.worksheets[0]
    rows, sect, seg, total = [], None, None, None
    for r in ws.iter_rows(values_only=True):
        r = list(r) + [None] * 16
        s1, s2 = r[1], r[2]
        if isinstance(s1, str) and s1.strip() in ('Domestic', 'Export'):
            sect = s1.strip()
            continue
        if sect != 'Domestic':
            continue
        if isinstance(s1, str) and s1.strip().startswith('Total'):
            if s1.strip() == 'Total':
                total = r[3:15]
            sect = 'after'
            continue
        if isinstance(s1, str) and s1.strip() in ('PC', 'RV', 'CV'):
            seg = s1.strip()
        if not isinstance(s2, str):
            continue
        for i, v in enumerate(r[3:15]):
            if isinstance(v, str):
                v = 0.0 if v.strip() in ('-', '') else float(v.replace(',', ''))
            rows.append(dict(month=f'{year}-{i + 1:02d}', segment=seg, name=s2.strip(), units=v))
    df = pd.DataFrame(rows)
    if total is None:
        raise ValueError(f'Domestic Total row not found in {path}')
    filled = {f'{year}-{i + 1:02d}' for i, v in enumerate(total) if isinstance(v, (int, float)) and v > 0}
    df = df[df.month.isin(filled)].copy()
    df['units'] = df.units.fillna(0.0).astype(float)
    df.attrs['total'] = {f'{year}-{i + 1:02d}': float(v) for i, v in enumerate(total) if isinstance(v, (int, float)) and v > 0}
    return df


# ------------------------------------------------------------------ KGM IR
def kgm_ir(path, year):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb[str(year)]
    rows, inside, total = [], False, None
    for r in ws.iter_rows(values_only=True):
        r = list(r) + [None] * 17
        if isinstance(r[0], str) and r[0].strip() == 'Domestic':
            inside = True
            continue
        if not inside:
            continue
        if isinstance(r[1], str) and 'Domestic Total' in r[1]:
            total = r[4:16]
            break
        if isinstance(r[2], str) and r[2].strip():
            for i, v in enumerate(r[4:16]):
                rows.append(dict(month=f'{year}-{i + 1:02d}', name=r[2].strip(), units=v))
    if total is None:
        raise ValueError(f'Domestic Total row not found in sheet {year} of {path}')
    filled = {f'{year}-{i + 1:02d}' for i, v in enumerate(total) if isinstance(v, (int, float)) and v > 0}
    df = pd.DataFrame(rows)
    df = df[df.month.isin(filled)].copy()
    df['units'] = pd.to_numeric(df.units, errors='coerce').fillna(0.0)
    df.attrs['total'] = {f'{year}-{i + 1:02d}': float(v) for i, v in enumerate(total) if isinstance(v, (int, float)) and v > 0}
    return df


# ------------------------------------------------------------------ HMG newsroom press (Hyundai / Kia, day 1)
def kor_num(s):
    """'5만 4,604' -> 54604; '7,427' -> 7427"""
    s = s.replace(',', '').strip()
    m = re.fullmatch(r'(?:(\d+)만\s*)?(\d*)', s)
    if not m:
        return None
    return float(int(m.group(1) or 0) * 10000 + int(m.group(2) or 0))


KNUM = r'((?:\d+만\s*)?\d{1,3}(?:,\d{3})*)'


def hmg_press(path):
    """-> dict(maker, month, models {name: units}, domestic_total, genesis_total) or None"""
    t = open(path, encoding='utf-8', errors='ignore').read()
    ti = re.search(r'<title>(.*?)</title>', t, re.S).group(1).strip()
    m = re.search(r'(현대차|기아),? (\d{4})년 (\d{1,2})월', ti)
    if not m:
        return None
    maker = 'Hyundai' if m.group(1) == '현대차' else 'Kia'
    month = f'{m.group(2)}-{int(m.group(3)):02d}'
    x = _text(t)
    i = x.find('■ 국내')
    j = x.find('■ 해외', i)
    seg = x[i:j] if i >= 0 else ''
    vals = {}
    for mm in re.finditer(r'([가-힣A-Za-z0-9Ⅲ]+(?: [0-9A-Za-z]+)?)(?:가|로|는)? (\d{1,3}(?:,\d{3})*)대', seg):
        nm = mm.group(1)
        if nm not in ('레이', '카니발', '니로'):
            nm = re.sub(r'(로|가|는)$', '', nm)
        if re.match(r'^\d+만$|^총$', nm):
            continue
        vals.setdefault(nm, float(mm.group(2).replace(',', '')))
    tot = re.search(r'국내(?:에서| 시장에서)[^다]{0,60}?' + KNUM + '대를? 판매', seg)
    gen = re.search(r'제네시스는[^.]*?총 ' + KNUM + '대', seg)
    return dict(maker=maker, month=month, models=vals,
                domestic_total=kor_num(tot.group(1)) if tot else None,
                genesis_total=kor_num(gen.group(1)) if gen else None)


# ------------------------------------------------------------------ KAIDA monthly release (brand totals)
KAIDA_BRAND = {'MINI': 'Mini'}


def kaida_press(path):
    x = _text(open(path, encoding='utf-8', errors='ignore').read())
    hd = re.search(r'(\d{1,2})월 수입 승용차 ([\d,]+)대 신규 ?등록 (\d{4})-(\d{2})-\d{2}', x)
    if not hd:
        raise ValueError(f'KAIDA headline not found in {path}')
    mon, y = int(hd.group(1)), int(hd.group(3))
    if mon > int(hd.group(4)):  # December figures published in January
        y -= 1
    month = f'{y}-{mon:02d}'
    i = x.find('브랜드별 등록대수')
    j = x.find('배기량별', i)
    brands = {}
    for m in re.finditer(r'([가-힣·\s]+)\(([^)]+)\)\s*([\d,]+)\s*대?', x[i:j]):
        b = m.group(2).strip()
        brands[KAIDA_BRAND.get(b, b)] = float(m.group(3).replace(',', ''))
    return month, brands, float(hd.group(2).replace(',', ''))
