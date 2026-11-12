"""Dataforce 유럽(EU+EFTA+UK) 상위 차종 표 읽기 — L'Automobile(ACI)·Razão Automóvel 기사 HTML.

parse_tables(html) -> [(header, rows)]  : 기사 안 <table> 마다 머리줄과 행(이름, 이번 값, 작년 값)
classify_table(header, month)            : 'month' | 'ytd' | 'year' | None  (그 달 표인지 누계 표인지)
to_ours(name, europe_list)               : Dataforce 이름 → 우리 Europe (Brand, Model) 또는 None
"""
import html as _html
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "europe_b"))

# 이름이 우리 표기와 다른 것(조사 때 확인한 10개 + 기사에서 본 것)
ALIAS = {"audi a3, s3": ("Audi", "A3"), "bmw serie 1": ("BMW", "1 Series"), "bmw série 1": ("BMW", "1 Series"),
         "citroën c3": ("Citroen", "C3"), "citroën c3 aircross": ("Citroen", "C3 Aircross"), "ford kuga": ("Ford", "Escape"),
         "mini one/cooper": ("MINI", "Cooper"), "opel corsa": ("Opel/Vauxhall", "Corsa"), "opel frontera": ("Opel/Vauxhall", "Frontera"),
         "renault 5 e-tech": ("Renault", "5"), "toyota corolla/ corolla ts": ("Toyota", "Corolla"), "toyota corolla/corolla ts": ("Toyota", "Corolla")}
MONTHS_IT = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"]
MONTHS_PT = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def _cells(row):
    return [_html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[dh][^>]*>.*?</t[dh]>", row, flags=re.S)]


def _num(s):
    s = re.sub(r"\s*(un\.?|unidades|units?)$", "", s.strip(), flags=re.I)        # Razão: '113 127 un.'
    s = s.replace("\xa0", "").replace(" ", "").replace(".", "").replace(",", "")
    return float(s) if re.fullmatch(r"\d+", s) else None


def parse_tables(page):
    out = []
    for tb in re.findall(r"<table.*?</table>", page, flags=re.S):
        rows = [_cells(r) for r in re.findall(r"<tr.*?</tr>", tb, flags=re.S)]
        if not rows:
            continue
        head, body = rows[0], []
        for r in rows[1:]:
            nums = [(i, _num(c)) for i, c in enumerate(r)]
            vals = [v for _, v in nums if v is not None]
            names = [c for c in r if _num(c) is None and not re.fullmatch(r"[+\-−]?[\d.,]+\s*%?", c) and c]
            if names and vals:
                # 첫 숫자가 순위면(1~50, 머리줄에 Pos 류) 건너뜀
                first_is_rank = bool(re.match(r"(pos|rank|posi)", head[0].lower())) if head else False
                v = vals[1:] if first_is_rank and len(vals) >= 2 else vals
                body.append((names[0], v[0], v[1] if len(v) > 1 else None))
        out.append((head, body))
    return out


def classify_table(head, month):
    """머리줄로 표 종류 판단: 그 달(예: 'Agosto 2026'), 누계(예: 'Gen.–ago. 2026', 'YTD 2026', 'Jan-Jun. 2026')."""
    h = " ".join(head).lower()
    y, m = month[:4], int(month[5:7])
    if y not in h:
        return None
    if re.search(r"ytd|gen\.?\s*[-–]|jan\.?\s*[-–/]|primi|semestre|jan-", h):
        return "ytd"
    if re.search(r"total\s*\d{4}", h):
        return "year"
    it, pt = MONTHS_IT[m - 1], MONTHS_PT[m - 1]
    full_it = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"][m - 1]
    full_pt = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"][m - 1]
    if re.search(rf"\b({it}|{pt}|{full_it}|{full_pt})\b", h):
        return "month"
    return None


def to_ours(name, europe_index):
    """europe_index: {(brand+' '+model).lower(): (Brand, Model)} — 우리 Europe 목록."""
    n = re.sub(r"\s+", " ", name.replace("Škoda", "Skoda").replace("ŠKODA", "Skoda")).strip()
    if n.lower() in ALIAS:
        return ALIAS[n.lower()]
    if n.lower() in europe_index:
        return europe_index[n.lower()]
    from europe_map import classify                  # 나라별 수집기와 같은 대응 규칙
    parts = n.split(" ", 1)
    b, mo, *_ = classify(parts[0], n)
    if b and mo and (b + " " + mo).lower() in europe_index:
        return europe_index[(b + " " + mo).lower()]
    return None


FULL_IT = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]
FULL_PT = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]


def _months_in(text):
    """글 안에 나온 달 번호들(앞에서부터). 약어(gen, ago, jun …)와 전체 이름 모두."""
    t = _html.unescape(text).lower()
    found = []
    for m in re.finditer(r"[a-zç]+", t):
        w = m.group(0)
        for i in range(12):
            if w in (MONTHS_IT[i], MONTHS_PT[i], FULL_IT[i], FULL_PT[i]):
                found.append(i + 1)
                break
    return found


def ytd_end(head):
    """누계 표 머리줄에서 누계가 몇 월까지인지(예: 'Gen.–ago. 2026' → 8). 달이 안 쓰였으면 None('YTD 2026')."""
    ms = _months_in(" ".join(c for c in head if re.search(r"\d{4}", c) and not c.lower().startswith(("var", "diff"))) or " ".join(head))
    return ms[-1] if ms and (len(ms) >= 2 or ms[-1] != 1) else (1 if ms == [1] else None)


def page_tables(page, year):
    """기사 하나의 표들 → [(kind, 월, 표 본문)]. kind = 'month' | 'ytd'.
    누계 표에 달이 안 쓰였으면 같은 기사 그 달 표의 달로 본다(ACI 5월 기사: 'Maggio 2026' + 'YTD 2026')."""
    tabs = parse_tables(page)
    out, article_month = [], None
    for head, body in tabs:
        cols = [_html.unescape(c) for c in head if str(year) in c and not c.lower().startswith(("var", "diff"))]  # 변화율 열 제외
        h = " ".join(cols).lower()
        if not cols or re.search(r"total\s*\d{4}", h):
            continue
        if re.search(r"ytd|[-–]|primi|semestre", h):
            out.append(["ytd", ytd_end(cols), body])
        else:
            ms = _months_in(h)
            if ms:
                out.append(["month", ms[0], body])
                article_month = article_month or ms[0]
    for row in out:
        if row[0] == "ytd" and row[1] is None:
            row[1] = article_month
    return [tuple(r) for r in out if r[1] is not None]
