"""Build the list of GCBC model-page URLs to fetch for Canada series (one row per series x candidate url).
Sources: group_1 gcbc_map.csv, coverage.csv match_name, gcbc_crawl.GUESS, and extra sitemap candidates
for the 50 Canada series that group_1 could not match (automatable == 'no')."""
import os, re, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
G1 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "data_sources_v1", "group_1"))
BASE = "https://www.goodcarbadcar.net/"

# extra candidates found by fuzzy search in the cached sitemap url lists (gcbc_urls.txt / gcbc_model_urls.txt)
EXTRA = {
    ("Land Rover", "Range Rover Sport"): ["land-rover-range-rover-sport-sales"],
    ("Land Rover", "Range Rover Evoque"): ["land-rover-range-rover-evoque-sales"],
    ("Mercedes-Benz", "AMG GT"): ["mercedes-amg-gt-sales-figures-usa-canada"],
    ("Kia", "K4"): ["kia-forte-sales-figures"],
    ("Subaru", "Crosstrek"): ["subaru-xv-crosstrek-sales-figures"],
    ("Mercedes-Benz", "GLC"): ["mercedes-benz-glc-class-sales-figures"],
    ("Mercedes-Benz", "GLE"): ["mercedes-benz-gle-class-sales-figures", "mercedes-benz-m-class-sales-figures"],
    ("Mercedes-Benz", "GLS"): ["mercedes-benz-gls-class-sales-figures", "mercedes-benz-gl-class-sales-figures"],
    ("Mercedes-Benz", "EQS"): ["mercedes-benz-eqs-sedan-sales-figures"],
    ("Lexus", "LC"): ["usa-canada-lexus-lc500-sales-stats-monthly-yearly"],
    ("Porsche", "Boxster"): ["porsche-718-sales-figures"],
    ("Porsche", "Cayman"): ["porsche-718-sales-figures"],
    ("Audi", "e-tron GT"): ["audi-e-tron-gt-sales-figures"],
    ("Lincoln", "Corsair"): ["lincoln-corsair-sales-figures", "lincoln-mkc-sales-figures-usa-canada"],
    ("Volvo", "EX30"): ["volvo-ex30-sales-figures"],
    ("Volvo", "EX90"): ["volvo-ex90-sales-figures"],
    ("Polestar", "3"): ["polestar-3-sales-figures"],
}


def main():
    ser = pd.read_csv(os.path.join(G1, "series.csv"), encoding="utf-8-sig")
    ca = ser[ser.Country == "Canada"][["Brand", "Model", "sales_last12"]]
    gmap = pd.read_csv(os.path.join(G1, "scripts", "gcbc_map.csv"), encoding="utf-8-sig")
    cov = pd.read_csv(os.path.join(G1, "coverage.csv"), encoding="utf-8-sig")
    cov = cov[cov.Country == "Canada"]
    sys.path.insert(0, os.path.join(G1, "scripts"))
    from gcbc_crawl import GUESS
    allurls = set(open(os.path.join(G1, "scripts", "gcbc_urls.txt")).read().split())
    rows = []
    for b, m, sl in ca.values:
        cand = []
        u = gmap[(gmap.Brand == b) & (gmap.Model == m)].gcbc_url.dropna().tolist()
        cand += [(x, "gcbc_map") for x in u]
        for note in cov[(cov.Brand == b) & (cov.Model == m)].note.fillna(""):
            for s in re.findall(r"\(([a-z0-9-]+-sales[a-z0-9-]*)\)", note):
                cand.append((BASE + s + "/", "coverage_note"))
        for mn in cov[(cov.Brand == b) & (cov.Model == m)].match_name.fillna(""):
            if re.match(r"^[a-z0-9-]+$", mn):
                cand.append((BASE + mn + "/", "coverage_match"))
        for g in GUESS.get((b, m), []):
            cand.append((BASE + g + "-sales-figures/", "group1_guess"))
        for g in EXTRA.get((b, m), []):
            cand.append((BASE + g + "/", "sitemap_fuzzy"))
        seen = set()
        for url, how in cand:
            if url in seen:
                continue
            seen.add(url)
            rows.append([b, m, sl, url, how, url in allurls])
        if not seen:
            rows.append([b, m, sl, "", "none", False])
    out = pd.DataFrame(rows, columns=["Brand", "Model", "sales_last12", "url", "how", "in_sitemap"])
    out = out.sort_values("sales_last12", ascending=False)
    out.to_csv(os.path.join(HERE, "ca_urls.csv"), index=False, encoding="utf-8-sig")
    print(len(out), out.url.nunique(), (out.url == "").sum())
    print(out.how.value_counts())


if __name__ == "__main__":
    main()
