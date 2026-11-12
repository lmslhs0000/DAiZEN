"""Collect US / Canada monthly model sales from GoodCarBadCar (GCBC) for given month(s).

Usage:
  python collect_na.py --month 2026-07                     # US + Canada, fetch if not cached
  python collect_na.py --month 2026-05,2026-06,2026-07 --country CA --offline
  python collect_na.py --month 2026-07 --refresh           # re-download cached pages (new month posted)
Options:
  --month     YYYY-MM, comma list allowed (each page is fetched once and all months are parsed from it)
  --country   US | CA | both (default both)
  --offline   use cached pages only (cache/ in this folder; group_1 sample for the 2026 US table)
  --refresh   ignore cache and re-download (polite: robots.txt, >=2.5 s per host, stop on HTTP 429)
  --out       output CSV (default out/collected_na_<first>_<last>.csv)
Output columns: Country,Month,Brand,Model,value,source_id,source_key,status
  status: OK | NOT_POSTED (cell blank on GCBC) | NOT_IN_TABLE | NO_CA_MONTHLY_TABLE | NO_URL | FETCH_FAIL
Also writes gcbc_monthly_long.csv (all parsed monthly values of the mapped series; used by split_rules.py).
Months > 2026-07 may be collected for format checks but must not be fed into our data (project rule)."""
import argparse, os, re, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gcbc_fetch  # noqa: E402
from gcbc_parse import parse_model_page, parse_us_table, page_title  # noqa: E402

G1 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "data_sources_v1", "group_1"))
BASE = "https://www.goodcarbadcar.net/"
US_TABLE_URL = BASE + "{year}-us-auto-sales-figures-by-model-all-vehicle-ranked/"

# our US (Brand, Model) -> list of GCBC US-table row names (several rows are summed: rule 'sum:a+b')
US_ALIAS = {
    ("Kia", "K4"): ["Kia K4 / Forte"],
    ("GMC", "Yukon / Yukon XL"): ["GMC Yukon"],
    ("Ford", "E-Series Van"): ["Ford E-Series"],
    ("Lincoln", "Nautilus"): ["Lincoln MKX/Nautilus"],
    ("Lincoln", "Corsair"): ["Lincoln Corsair / MKC"],
    ("Mercedes-Benz", "E-Class / CLS"): ["Mercedes-Benz E-Class"],
    ("Land Rover", "Discovery"): ["Land Rover Discovery / LR4"],
    ("Rolls-Royce", "Cullinan"): ["Rolls Royce Cullinan"],
    ("Polestar", "3"): ["Polestar Polestar 3"],
    ("Polestar", "4"): ["Polestar Polestar 4"],
    ("Volkswagen", "Atlas"): ["Volkswagen Atlas", "Volkswagen Atlas Sport"],  # quarter sums equal ours (Atlas + Cross Sport)
}
# our Canada (Brand, Model) -> GCBC page slug(s) to try first (verified pages with a Canada monthly table)
CA_URL_OVERRIDE = {}


def norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def series_list():
    s = pd.read_csv(os.path.join(G1, "series.csv"), encoding="utf-8-sig")
    return s[["Country", "Brand", "Model", "sales_last12"]]


def load_page(url, offline, refresh):
    try:
        st, html = gcbc_fetch.get(url, offline=offline, refresh=refresh)
    except RuntimeError as e:  # HTTP 429 -> stop host
        print("STOP:", e)
        raise
    return st, html


def us_table(year, offline, refresh):
    url = US_TABLE_URL.format(year=year)
    st, html = load_page(url, offline, refresh)
    if not html and year == 2026:
        p = os.path.join(G1, "samples", "gcbc_us_2026_by_model.html")
        if os.path.exists(p):
            st, html = "group1_sample", open(p, encoding="utf-8").read()
    return url, st, (parse_us_table(html, year) if html else {})


def collect_us(months, offline, refresh, long_rows):
    ser = series_list()
    ser = ser[ser.Country == "United States"]
    out = []
    tabs = {}
    for y in sorted({int(m[:4]) for m in months}):
        tabs[y] = us_table(y, offline, refresh)
    gmap = pd.read_csv(os.path.join(G1, "scripts", "gcbc_map.csv"), encoding="utf-8-sig")
    for b, m, _ in ser[["Brand", "Model", "sales_last12"]].values:
        for mo in months:
            url, st, tab = tabs[int(mo[:4])]
            keys = {norm(k): k for k in tab}
            names = US_ALIAS.get((b, m))
            if names is None:
                k = keys.get(norm(b + m))
                names = [k] if k else []
            names = [n for n in names if n in tab]
            if names:
                vals = [tab[n].get(mo) for n in names]
                key = "+".join(names)
                if any(v is None for v in vals):
                    out.append(["United States", mo, b, m, None, "GCBC_US_TABLE", key, "NOT_POSTED"])
                else:
                    out.append(["United States", mo, b, m, sum(vals), "GCBC_US_TABLE", key, "OK"])
                for n in names:
                    for per, v in tab[n].items():
                        long_rows.append(["United States", b, m, url, "M", per, v, n])
                continue
            # fallback: GCBC model page, US monthly table
            u = gmap[(gmap.Brand == b) & (gmap.Model == m)].gcbc_url.dropna().tolist()
            if not u:
                out.append(["United States", mo, b, m, None, "GCBC", "", "NOT_IN_TABLE"])
                continue
            stp, html = load_page(u[0], offline, refresh)
            rows = [r for r in parse_model_page(html) if r[0] == "United States" and r[1] == "M"] if html else []
            d = {r[2]: r[3] for r in rows}
            slug = u[0].replace(BASE, "").strip("/")
            if mo in d:
                v = d[mo]
                out.append(["United States", mo, b, m, v, "GCBC", slug, "OK" if v is not None else "NOT_POSTED"])
            else:
                out.append(["United States", mo, b, m, None, "GCBC", slug, "NOT_IN_TABLE" if html else "FETCH_FAIL"])
    return out


def zero_placeholder(d, mo):
    """GCBC Canada tables show 0 for months of quarter-reporting brands that are not estimated yet
    (e.g. Corvette / Tesla Model S 2026-07..09 = 0 after 290 / 24 in June). Flag a 0 when it starts a run
    of at most 3 zeros (latest quarter) up to the last posted month and the 3 months before average >= 10 units.
    Longer zero runs (e.g. Mazda CX-50 2026-05..09, ours also 0) are kept as real zeros."""
    y = mo[:4]
    cutoff = str(pd.Period.now("M") - 1)  # GCBC fills future months of the year with 0 on some pages
    later = [v for k, v in d.items() if k[:4] == y and mo <= k <= cutoff and v is not None]
    prev = [d.get(str(pd.Period(mo) - i)) for i in (1, 2, 3)]
    prev = [p for p in prev if p is not None]
    return bool(later) and len(later) <= 3 and all(v == 0 for v in later) and len(prev) == 3 and sum(prev) / 3 >= 10


def ca_candidates():
    p = os.path.join(HERE, "ca_urls.csv")
    u = pd.read_csv(p, encoding="utf-8-sig")
    cand = {}
    for b, m, url in u[["Brand", "Model", "url"]].fillna("").values:
        if url:
            cand.setdefault((b, m), []).append(url)
    for k, slugs in CA_URL_OVERRIDE.items():
        cand[k] = [BASE + s + "/" for s in slugs] + [x for x in cand.get(k, []) if x.replace(BASE, "").strip("/") not in slugs]
    return cand


def collect_ca(months, offline, refresh, long_rows):
    ser = series_list()
    ser = ser[ser.Country == "Canada"].sort_values("sales_last12", ascending=False)
    cand = ca_candidates()
    parsed = {}
    out = []
    for b, m, _ in ser[["Brand", "Model", "sales_last12"]].values:
        urls = cand.get((b, m), [])
        chosen, d, annual_only, fail = None, {}, False, False
        for url in urls:
            if url not in parsed:
                st, html = load_page(url, offline, refresh)
                parsed[url] = (st, parse_model_page(html) if html else None,
                               page_title(html) if html else "")
            st, rows, title = parsed[url]
            if rows is None:
                fail = True
                continue
            if title.startswith("Automotive Sales Figures"):  # soft 404 -> home page
                continue
            cm = {r[2]: r[3] for r in rows if r[0] == "Canada" and r[1] == "M"}
            if cm:
                chosen, d = url, cm
                break
            if any(r[0] == "Canada" and r[1] == "Y" for r in rows):
                annual_only = True
        if chosen:
            slug = chosen.replace(BASE, "").strip("/")
            for mo in months:
                v = d.get(mo)
                st = "OK" if v is not None else ("NOT_POSTED" if mo in d else "NOT_IN_TABLE")
                if v == 0 and zero_placeholder(d, mo):
                    v, st = None, "ZERO_PLACEHOLDER"
                out.append(["Canada", mo, b, m, v, "GCBC", slug, st])
            for per, v in d.items():
                long_rows.append(["Canada", b, m, chosen, "M", per, v, slug])
        else:
            st = "NO_URL" if not urls else ("FETCH_FAIL" if fail else "NO_CA_MONTHLY_TABLE")
            key = ";".join(x.replace(BASE, "").strip("/") for x in urls)
            if annual_only:
                key += " (Canada annual table only)"
            for mo in months:
                out.append(["Canada", mo, b, m, None, "GCBC", key, st])
    # US monthly values from all parsed Canada-candidate pages (for US quarterly shape tests)
    for (b, m), urls in cand.items():
        for url in urls:
            st, rows, title = parsed.get(url, (None, None, ""))
            if rows:
                for r in rows:
                    if r[0] == "United States" and r[1] == "M":
                        long_rows.append(["United States(page)", b, m, url, "M", r[2], r[3], url.replace(BASE, "").strip("/")])
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, action="append")
    ap.add_argument("--country", default="both", choices=["US", "CA", "both"])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    months = sorted({x.strip() for m in a.month for x in m.split(",") if x.strip()})
    for mo in months:
        if not re.match(r"^\d{4}-\d{2}$", mo):
            sys.exit(f"bad month {mo}")
    long_rows, out = [], []
    if a.country in ("US", "both"):
        out += collect_us(months, a.offline, a.refresh, long_rows)
    if a.country in ("CA", "both"):
        out += collect_ca(months, a.offline, a.refresh, long_rows)
    df = pd.DataFrame(out, columns=["Country", "Month", "Brand", "Model", "value", "source_id", "source_key", "status"])
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    path = a.out or os.path.join(HERE, "out", f"collected_na_{months[0]}_{months[-1]}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    lg = pd.DataFrame(long_rows, columns=["Country", "Brand", "Model", "url", "kind", "period", "value", "source_key"])
    lg["Country"] = lg.Country.replace({"United States(page)": "United States"})
    lg = lg.drop_duplicates(["Country", "Brand", "Model", "period"], keep="first")
    lg.to_csv(os.path.join(HERE, "gcbc_monthly_long.csv"), index=False, encoding="utf-8-sig")
    print(path)
    print(df.groupby(["Country", "Month", "status"]).size().to_string())


if __name__ == "__main__":
    main()
