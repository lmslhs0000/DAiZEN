"""China monthly collector: Gasgoo CPCA model pages -> our 677 China series (CPCA wholesale basis, exports included).

usage:
  python collect_china.py 2026 9                 # collect month 2026-09 (run around the end of the next month)
  python collect_china.py 2026 7 --upto 2026-08  # also allow later pages (M+1..M+2) to read month M (back-fill / validation)
  python collect_china.py 2026 7 --no-fetch      # cache only
options: --mapping <csv> (default mapping.csv here), --out <dir> (default out/)

What it does
  1. reads mapping.csv (Country, Brand, Model, source_id, source_key, rule, confidence, note).
     source_key = CPCA line slugs joined by '+' (sum). rule = sum | split:<rule_id> (split_rules.csv) | none.
  2. for every line fetches the model page /autosales/model/<slug>/<y>/<m>.html (cache first, >= 2.5 s between live
     requests, HTTP 429 -> stop). A page for month M shows the line's own row only when the line's M value is positive;
     otherwise the first row is the OEM total (trap). So: row present -> value; row absent -> 'not_positive' (0 or not
     reported; NOT written as 0). With --upto later pages are also tried (they repeat months M-2..M).
  3. series value = sum of its lines (lines with 'not_positive' count as 0 only if at least one line of the series is
     positive; otherwise the series value is blank with status not_positive). Split rules give an allocation (marked).
  4. checks: month-over-month jump (x5 or /5 vs previous month on the same page, both > 200) -> flag 'jump';
     line value (> 300) equal to another line's value in the same month (copy error) -> flag 'dup_value'.
Output: out/collected_<y>_<m>.csv (series level), out/lines_<y>_<m>.csv (line level, all 3 months on each page so the
previous two months can be re-checked for CPCA revisions)."""
import sys, argparse
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gasgoo


def prev_month(month, k=1):
    y, m = int(month[:4]), int(month[5:])
    t = y * 12 + m - 1 - k
    return f"{t // 12}-{t % 12 + 1:02d}"


def collect(y, m, mapping, split_rules, fetch=True, upto=None, months=None):
    """-> (series DataFrame, lines DataFrame) for target month (and optional extra months via `months`)."""
    target = f"{y}-{m:02d}"
    months = months or [target]
    upto = upto or target
    lines = {}
    for key in mapping.source_key.dropna():
        for s in str(key).split("+"):
            s = s.strip()
            if s:
                lines.setdefault(s, None)
    lrows = []
    for s in lines:
        for mo in months:
            v, st, url = gasgoo.month_value(s, mo, fetch=fetch, max_ahead=2, upto=upto)
            prev, pst, _ = gasgoo.month_value(s, prev_month(mo), fetch=False, max_ahead=2, upto=upto)
            lrows.append(dict(slug=s, Month=mo, value=v, status=st, prev_value=prev, url=url))
    L = pd.DataFrame(lrows)
    # copy-error check: identical value in two different lines in the same month (both > 300; coincidences below that are common)
    dup = L[(L.value.fillna(0) > 300)].groupby(["Month", "value"]).slug.transform("count") > 1
    L["dup_value"] = False
    L.loc[dup[dup].index, "dup_value"] = True
    lv = L.set_index(["slug", "Month"])
    srows = []
    sr = split_rules.set_index("rule_id") if len(split_rules) else None
    for _, r in mapping.iterrows():
        for mo in months:
            base = dict(Country=r.Country, Brand=r.Brand, Model=r.Model, Month=mo, rule=r.rule, source_key=r.source_key,
                        confidence=r.confidence)
            if not isinstance(r.source_key, str) or not r.source_key.strip() or r.rule == "none":
                srows.append(dict(base, value=None, status="no_source", flags="", urls=""))
                continue
            comps = [s.strip() for s in r.source_key.split("+") if s.strip()]
            vals, sts, urls, flags = [], [], [], []
            for s in comps:
                x = lv.loc[(s, mo)]
                vals.append(x.value); sts.append(x.status); urls.append(x.url)
                if x.dup_value:
                    flags.append(f"dup_value:{s}")
                if pd.notna(x.value) and pd.notna(x.prev_value) and max(x.value, x.prev_value) > 200 and \
                        (x.value > 5 * max(x.prev_value, 1) or x.prev_value > 5 * max(x.value, 1)):
                    flags.append(f"jump:{s}:{int(x.prev_value)}->{int(x.value)}")
            pos = [v for v in vals if pd.notna(v)]
            if "missing" in sts:
                value, status = None, "missing_page"
            elif pos:
                value = float(sum(pos))
                status = "ok" if all(st == "ok" for st in sts) else "ok_some_lines_not_positive"
            else:
                value, status = None, "not_positive"
            if isinstance(r.rule, str) and r.rule.startswith("split:") and value is not None:
                rid = r.rule.split(":", 1)[1]
                share = float(sr.loc[rid, "share"]) if r.Model == sr.loc[rid, "series_a"] else 1 - float(sr.loc[rid, "share"])
                value = round(value * share, 1)
                status += ";allocation"
            srows.append(dict(base, value=value, status=status, flags=";".join(flags), urls=" | ".join(urls)))
    return pd.DataFrame(srows), L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("month", type=int)
    ap.add_argument("--upto", default=None, help="last page month allowed, YYYY-MM (default = target month)")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--mapping", default=str(HERE / "mapping.csv"))
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args()
    mp = pd.read_csv(a.mapping, encoding="utf-8-sig")
    sp = HERE / "split_rules.csv"
    split_rules = pd.read_csv(sp, encoding="utf-8-sig") if sp.exists() else pd.DataFrame()
    S, L = collect(a.year, a.month, mp, split_rules, fetch=not a.no_fetch, upto=a.upto)
    out = Path(a.out); out.mkdir(exist_ok=True)
    tag = f"{a.year}_{a.month}"
    S.to_csv(out / f"collected_{tag}.csv", index=False, encoding="utf-8-sig")
    L.to_csv(out / f"lines_{tag}.csv", index=False, encoding="utf-8-sig")
    print(S.status.value_counts().to_string())
    print("requests:", gasgoo.STATS)


if __name__ == "__main__":
    main()
