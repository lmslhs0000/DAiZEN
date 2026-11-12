"""Build validation_na.csv, mapping_na.csv, resolved_no_ca.csv and match-rate summary from a collect_na.py output.
Usage: python build_outputs_na.py out/collected_na_2026-05_2026-07.csv"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
G1 = os.path.join(ROOT, "experiments", "data_sources_v1", "group_1")
MONTHS = ["2026-05", "2026-06", "2026-07"]


def main(path):
    col = pd.read_csv(path, encoding="utf-8-sig")
    col = col[col.Month.isin(MONTHS)]
    raw = pd.read_csv(os.path.join(ROOT, "raw", "data.csv"), encoding="utf-8-sig")
    raw = raw[raw.Country.isin(["United States", "Canada"]) & raw.Month.isin(MONTHS)]
    ours = {(r.Country, r.Brand, r.Model, r.Month): r.Sales for r in raw.itertuples()}
    cov = pd.read_csv(os.path.join(G1, "coverage.csv"), encoding="utf-8-sig")

    rows = []
    for r in col.itertuples():
        o = ours.get((r.Country, r.Brand, r.Model, r.Month), np.nan)
        v = r.value
        if pd.isna(v) or pd.isna(o):
            st, diff = "MISSING", np.nan
        else:
            diff = v - o
            st = "EQUAL" if abs(diff) < 0.5 else "DIFF"
        rows.append([r.Country, r.Brand, r.Model, r.Month, o, v, diff, st, r.source_id, r.source_key, r.status])
    val = pd.DataFrame(rows, columns=["Country", "Brand", "Model", "Month", "ours", "collected", "diff", "status",
                                      "source_id", "source_key", "collect_status"])
    corr = pd.read_csv(os.path.join(ROOT, "raw", "data_corrections.csv"), encoding="utf-8-sig")
    cset = set(zip(corr.Country, corr.Month, corr.Brand, corr.Model))
    val["ours_corrected_in_raw"] = [(c, mo, b, m) in cset for c, mo, b, m in zip(val.Country, val.Month, val.Brand, val.Model)]
    val.to_csv(os.path.join(HERE, "validation_na.csv"), index=False, encoding="utf-8-sig")

    # match rates
    summ = []
    for c, d in list(val.groupby("Country")) + [("ALL", val)]:
        for scope, dd in [("all rows", d), ("posted rows (excl. MISSING)", d[d.status != "MISSING"])]:
            w = dd.ours.fillna(0)
            n = len(dd)
            summ.append([c, scope, n, (dd.status == "EQUAL").sum(), (dd.status == "DIFF").sum(),
                         (dd.status == "MISSING").sum(),
                         round(100 * (dd.status == "EQUAL").mean(), 2) if n else np.nan,
                         round(100 * w[dd.status == "EQUAL"].sum() / w.sum(), 2) if w.sum() else np.nan])
    for c, d in val.groupby("Country"):
        for mo, dd in d.groupby("Month"):
            w = dd.ours.fillna(0)
            summ.append([c, f"month {mo}", len(dd), (dd.status == "EQUAL").sum(), (dd.status == "DIFF").sum(),
                         (dd.status == "MISSING").sum(), round(100 * (dd.status == "EQUAL").mean(), 2),
                         round(100 * w[dd.status == "EQUAL"].sum() / w.sum(), 2)])
    summ = pd.DataFrame(summ, columns=["Country", "scope", "n", "equal", "diff", "missing", "equal_pct_count",
                                       "equal_pct_sales"])
    summ.to_csv(os.path.join(HERE, "validation_na_summary.csv"), index=False, encoding="utf-8-sig")
    print(summ.to_string())

    # mapping
    sr = pd.read_csv(os.path.join(HERE, "split_rules.csv"), encoding="utf-8-sig")
    srd = {(r.Country, r.Brand, r.Model): r for r in sr.itertuples()}
    mp = []
    for (c, b, m), d in val.groupby(["Country", "Brand", "Model"], sort=False):
        cv = cov[(cov.Country == c) & (cov.Brand == b) & (cov.Model == m)].iloc[0]
        key = d.source_key.dropna().iloc[0] if d.source_key.notna().any() else ""
        cst = set(d.collect_status)
        neq, ndiff, nmiss = (d.status == "EQUAL").sum(), (d.status == "DIFF").sum(), (d.status == "MISSING").sum()
        gc_ok = bool(cst & {"OK", "NOT_POSTED"})
        notes = [f"2026-05..07 check: equal={neq} diff={ndiff} missing={nmiss}"]
        if gc_ok:
            sid = d.source_id.iloc[0]
            rule = f"sum:{key}" if "+" in str(key) else "direct"
            conf = "high" if neq >= 2 and ndiff == 0 else ("medium" if neq >= 1 else "low")
            if "NOT_POSTED" in cst:
                notes.append("GCBC posts this model late (quarter-reporting brand): blank months until the OEM quarter")
        else:
            sid, rule, conf = "", "none", "low"
            notes.append(f"GCBC: {'/'.join(sorted(cst))} {key}".strip())
        if (c, b, m) in srd:
            s = srd[(c, b, m)]
            notes.append(f"OEM {s.source_id} {s.period_type}: {s.rule} (backtest WAPE {s.backtest_wape}%)")
            if not gc_ok:
                sid, rule, key, conf = s.source_id, s.rule, s.source_id, "medium"
        notes.append(f"group_1 coverage: {cv.automatable} / {cv.source_id}")
        mp.append([c, b, m, sid, key, rule, conf, "; ".join(notes), cv.sales_last12])
    mp = pd.DataFrame(mp, columns=["Country", "Brand", "Model", "source_id", "source_key", "rule", "confidence", "note",
                                   "sales_last12"]).sort_values(["Country", "sales_last12"], ascending=[False, False])
    mp.to_csv(os.path.join(HERE, "mapping_na.csv"), index=False, encoding="utf-8-sig")
    print(mp.groupby(["Country", "rule"]).sales_last12.agg(["size", "sum"]).to_string())

    # the 50 Canada series that group_1 could not match
    no = cov[(cov.Country == "Canada") & (cov.automatable == "no")]
    res = []
    for r in no.itertuples():
        d = val[(val.Country == "Canada") & (val.Brand == r.Brand) & (val.Model == r.Model)]
        cst = set(d.collect_status)
        resolved = bool(cst & {"OK", "NOT_POSTED"})
        res.append([r.Brand, r.Model, r.sales_last12, resolved, d.source_key.iloc[0] if len(d) else "",
                    (d.status == "EQUAL").sum(), (d.status == "DIFF").sum(), (d.status == "MISSING").sum(),
                    "/".join(sorted(cst))])
    res = pd.DataFrame(res, columns=["Brand", "Model", "sales_last12", "resolved", "source_key", "equal", "diff",
                                     "missing", "collect_status"])
    res.to_csv(os.path.join(HERE, "resolved_no_ca.csv"), index=False, encoding="utf-8-sig")
    print("resolved 'no' Canada series:", res.resolved.sum(), "/", len(res),
          f"sales share {100 * res[res.resolved].sales_last12.sum() / res.sales_last12.sum():.1f}%")


QREP_US = {"Chevrolet", "GMC", "Buick", "Cadillac", "Jeep", "Ram", "Dodge", "Chrysler", "Fiat", "Alfa Romeo",
           "Nissan", "Infiniti", "BMW", "Mini", "Mitsubishi"}
# official US Q3-2026 lines in group_1 sample texts (BMW chart 1, Stellantis summary): our model -> line names
OEM_Q3 = {
    ("BMW", "2 Series"): ["2 Series"], ("BMW", "7 Series"): ["7 Series"], ("BMW", "8 Series"): ["8 Series"],
    ("BMW", "X2"): ["X2"], ("BMW", "X4"): ["X4"], ("BMW", "X6"): ["X6"], ("BMW", "XM"): ["XM"], ("BMW", "Z4"): ["Z4"],
    ("BMW", "iX"): ["iX"], ("Mini", "Countryman"): ["Countryman"],
    ("Mini", "Cooper"): ["Cooper /S Hardtop 2 Door", "Cooper /S Hardtop 4 Door", "Cooper /S Convertible", "Cooper /S Clubman"],
    ("Alfa Romeo", "Giulia"): ["Giulia"], ("Alfa Romeo", "Stelvio"): ["Stelvio"], ("Alfa Romeo", "Tonale"): ["Tonale"],
    ("Dodge", "Hornet"): ["Hornet"], ("Dodge", "Charger"): ["Charger ICE"], ("Dodge", "Charger BEV"): ["Charger BEV"],
    ("Jeep", "Wagoneer S"): ["Wagoneer S"], ("Jeep", "Recon"): ["Recon"], ("Fiat", "500"): ["500e"], ("Fiat", "500X"): ["500X"],
}


def oem_q3_lines():
    import re
    st = os.path.join(G1, "scripts", "sample_text")
    d = {}
    for f in ["us_bmw_2026Q3_chart1.pdf.txt", "us_stellantis_2026Q3.pdf.txt"]:
        for line in open(os.path.join(st, f), encoding="utf-8"):
            m = re.match(r"^(.+?) ([\d,]+) ([\d,]+) -?[\d.,]+% ([\d,]+) ([\d,]+)", line.strip())
            if m:
                d[(f, m.group(1))] = int(m.group(2).replace(",", ""))
    out = {}
    for k, names in OEM_Q3.items():
        vals = [v for (f, n), v in d.items() if n in names]
        if len(vals) == len(names):
            src = "BMW NA Q3-2026 chart (samples/us_bmw_2026Q3_chart1.pdf)" if k[0] in ("BMW", "Mini") else \
                "Stellantis US Q3-2026 summary (samples/us_stellantis_2026Q3.pdf)"
            out[k] = (sum(vals), src)
    return out


def proposed_fixes(val):
    """Only clear cases: a 2026-07 value of a quarterly-reporting brand that is GCBC's pre-release estimate,
    now replaced by GCBC's split of the official Q3 total (same convention the main already applied to 54 rows
    in raw/data_corrections.csv). Requires: GCBC Jul+Aug+Sep posted; where an OEM Q3 line is available it must
    equal the GCBC Q3 sum. Rows already in data_corrections.csv are skipped."""
    sys.path.insert(0, HERE)
    import collect_na
    url, st, tab = collect_na.us_table(2026, offline=True, refresh=False)
    corr = pd.read_csv(os.path.join(ROOT, "raw", "data_corrections.csv"), encoding="utf-8-sig")
    done = set(zip(corr.Country, corr.Month, corr.Brand, corr.Model))
    oem = oem_q3_lines()
    fx, skipped = [], []
    d = val[(val.Country == "United States") & (val.Month == "2026-07") & (val.status == "DIFF")]
    for r in d.itertuples():
        if (r.Country, r.Month, r.Brand, r.Model) in done:
            continue
        if r.Brand not in QREP_US:
            skipped.append([r.Brand, r.Model, "brand reports monthly / not quarter-only"])
            continue
        names = str(r.source_key).split("+")
        q3 = [tab.get(n, {}).get(m) for n in names for m in ("2026-07", "2026-08", "2026-09")]
        if any(v is None for v in q3):
            skipped.append([r.Brand, r.Model, "GCBC Q3 not fully posted"])
            continue
        gq3 = sum(q3)
        ev = f"GCBC US 2026 table (fetched 2026-10-14/15) Jul={r.collected:.0f}, Jul+Aug+Sep={gq3:.0f}"
        o = oem.get((r.Brand, r.Model))
        if o:
            if abs(o[0] - gq3) > 0.5:
                skipped.append([r.Brand, r.Model, f"GCBC Q3 {gq3} != OEM Q3 {o[0]}"])
                continue
            ev += f"; equals official Q3-2026 total {o[0]} in {o[1]}"
            if r.ours > o[0]:
                ev += f"; our July {r.ours:.0f} alone exceeds the official Q3 total"
        else:
            ev += "; OEM Q3 model line not checked (GM/Nissan/Infiniti/Mitsubishi sample not parsed or not public)"
        fx.append(["United States", "2026-07", r.Brand, r.Model, r.ours, r.collected,
                   "quarterly-reporting brand: our 2026-07 value was GoodCarBadCar's provisional monthly estimate made "
                   "before the Q3 company release; replace with GCBC's revised monthly value (same rule as the 54 rows "
                   "applied on 2026-10-07)", ev + "; https://www.goodcarbadcar.net/2026-us-auto-sales-figures-by-model-all-vehicle-ranked/"])
    return fx, skipped


if __name__ == "__main__":
    main(sys.argv[1])
    v = pd.read_csv(os.path.join(HERE, "validation_na.csv"), encoding="utf-8-sig")
    fx, sk = proposed_fixes(v)
    cols = ["Country", "Month", "Brand", "Model", "old_sales", "new_sales", "reason", "evidence"]
    pd.DataFrame(fx, columns=cols).to_csv(os.path.join(HERE, "proposed_fixes_na.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(sk, columns=["Brand", "Model", "why_not_proposed"]).to_csv(
        os.path.join(HERE, "proposed_fixes_na_skipped.csv"), index=False, encoding="utf-8-sig")
    print("proposed fixes:", len(fx), "skipped:", len(sk))
