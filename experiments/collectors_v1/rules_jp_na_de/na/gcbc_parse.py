"""Parsers for GoodCarBadCar pages.
- parse_model_page(html): monthly/annual tables on a model page -> list of (region, kind, period, value)
  region in {United States, Canada, Europe, China, Mexico, ''}; kind 'M' (period YYYY-MM) or 'Y' (YYYY).
  A blank cell (month not yet posted) gives value None.
- parse_us_table(html): the 'all models' yearly page -> {row name: {YYYY-MM: value or None}}"""
import re
from bs4 import BeautifulSoup

MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _num(x):
    x = x.replace(",", "").strip()
    if x in ("", "-", "–", "—", "N/A", "n/a"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _region_of(t):
    for h in t.find_all_previous(["h1", "h2", "h3", "h4", "h5"]):
        x = h.get_text(" ", strip=True)
        m = re.search(r"(United States|U\.S\.?|USA|Canada|Canadian|Europe|China|Mexico)", x)
        if m and re.search(r"Sales", x, re.I):
            r = m.group(1)
            if r.startswith("U"):
                return "United States"
            return "Canada" if r.startswith("Canad") else r
    return ""


def page_title(html):
    s = BeautifulSoup(html, "lxml")
    return s.title.get_text(strip=True) if s.title else ""


def parse_model_page(html):
    s = BeautifulSoup(html, "lxml")
    rows = []
    for t in s.find_all("table"):
        tr = t.find("tr")
        if not tr:
            continue
        hdr = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        reg = _region_of(t)
        if hdr[:2] == ["Year", "Jan"]:
            for r in t.find_all("tr")[1:]:
                c = [x.get_text(" ", strip=True) for x in r.find_all(["td", "th"])]
                if c and re.match(r"^\d{4}$", c[0]):
                    for i in range(12):
                        v = _num(c[i + 1]) if i + 1 < len(c) else None
                        rows.append((reg, "M", f"{c[0]}-{i+1:02d}", v))
        elif hdr[:2] in (["Year", "Units Sold"], ["Year", "Units"], ["Year", "Sales Units"], ["Year", "Sales"]):
            for r in t.find_all("tr")[1:]:
                c = [x.get_text(" ", strip=True) for x in r.find_all(["td", "th"])]
                if c and re.match(r"^\d{4}", c[0]):
                    rows.append((reg, "Y", c[0][:4], _num(c[1]) if len(c) > 1 else None))
    return rows


def parse_us_table(html, year):
    s = BeautifulSoup(html, "lxml")
    for t in s.find_all("table"):
        tr = t.find("tr")
        hdr = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])] if tr else []
        if hdr[:3] == ["Model", "Jan", "Feb"]:
            tab = {}
            for r in t.find_all("tr")[1:]:
                c = [x.get_text(" ", strip=True) for x in r.find_all(["td", "th"])]
                if len(c) >= 13:
                    tab[c[0]] = {f"{year}-{i+1:02d}": _num(c[i + 1]) for i in range(12)}
            return tab
    return {}
