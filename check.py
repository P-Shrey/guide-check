"""Guide Check without the front end.

Runs the same scoring the web app runs (src/serve.py) on your own inputs.

One property:

    python check.py --suburb PARRAMATTA --postcode 2150 --type house --land 450 --guide 950000

Many properties, from a CSV with columns suburb, postcode, property_type,
land_area_m2 (may be blank) and guide; any other columns are kept:

    python check.py --csv examples/demo-inputs.csv
    python check.py --csv my-cases.csv --out my-results.csv

The first run takes 30-45 seconds while the model loads; each check after
that takes about a second.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import serve  # noqa: E402

FIELDS = ["severity", "band", "comps_percentile", "gap_vs_comps_pct", "gap_in_typical_errors",
          "comps_estimate", "comps_n", "point", "lower", "upper", "typical_error_pct", "verdict"]


def check(suburb, postcode, property_type, land, guide):
    return serve.value_manual(suburb=suburb, postcode=postcode or "0000",
                              property_type=property_type or "house",
                              land_area_m2=float(land) if land not in (None, "") else None,
                              guide=float(guide))


def print_report(r):
    s = r["subject"]
    land = f"{s['land_area_m2']:.0f} m2" if s.get("land_area_m2") else "no land area"
    print(f"\n{s['suburb']} {s['postcode']}, {s['property_type']}, {land}, "
          f"guide ${r['guide']:,.0f} (valued as at {s['valued_as_at']})\n")
    print(f"  Result:      {r['severity'].upper()}: {r.get('band', '')}")
    print(f"  {r['verdict']}\n")
    if r.get("comps_n"):
        print(f"  Comparable-sales estimate  ${r['comps_estimate']:,.0f}  (median of {r['comps_n']} adjusted sales)")
        print(f"  Guide percentile           {r['comps_percentile']:.0f}th")
        print(f"  Gap to the comparables     {r['gap_vs_comps_pct']:.1f}% below "
              f"= {r['gap_in_typical_errors']:.2f} typical errors")
    print(f"  Model estimate             ${r['point']:,.0f}  "
          f"(80% interval ${r['lower']:,.0f} to ${r['upper']:,.0f}; measured coverage "
          f"{r['empirical_coverage_pct']}%)")

    comps = r.get("comps")
    if comps is not None and len(comps):
        print("\n  Nearest comparable sales (sold price, then adjusted to this property):")
        for _, c in comps.iterrows():
            print(f"    {str(c.get('address', '')):<28} {str(c.get('suburb', '')):<16} "
                  f"{str(c.get('contract_date', ''))[:10]}  ${c['sale_price']:>11,.0f}  "
                  f"-> ${c['adjusted_price']:>11,.0f}")

    print("\n  Largest drivers of the model estimate:")
    for d in r.get("top_drivers", []):
        print(f"    {d['feature']:<24} {str(d['value']):<14} {d['effect_pct']:+.1f}%")
    print(f"\n  {r['disclaimer']}\n")


def run_csv(path, out):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if not rows:
        sys.exit(f"{path} has no rows")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + FIELDS + ["error"])
        w.writeheader()
        for i, row in enumerate(rows, 1):
            try:
                r = check(row["suburb"], row.get("postcode"), row.get("property_type"),
                          row.get("land_area_m2"), row["guide"])
                result = {k: r.get(k) for k in FIELDS}
                print(f"  {i:>3}/{len(rows)}  {row['suburb']:<20} ${float(row['guide']):>12,.0f}  "
                      f"-> {r['severity']}")
            except (ValueError, KeyError) as e:  # bad suburb or missing column: record it, keep going
                result = {"error": str(e)}
                print(f"  {i:>3}/{len(rows)}  {row.get('suburb', '?'):<20} ERROR {e}")
            w.writerow({**row, **result})
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Check an advertised NSW price guide against comparable sales.")
    ap.add_argument("--suburb"); ap.add_argument("--postcode", default="")
    ap.add_argument("--type", default="house", choices=["house", "unit"])
    ap.add_argument("--land", help="land area in m2 (leave out for units)")
    ap.add_argument("--guide", type=float, help="advertised guide in dollars, e.g. 950000")
    ap.add_argument("--csv", help="score every row of this CSV instead")
    ap.add_argument("--out", help="where to write CSV results (default: <input>-results.csv)")
    a = ap.parse_args()

    if a.csv:
        run_csv(a.csv, a.out or str(Path(a.csv).with_name(Path(a.csv).stem + "-results.csv")))
    elif a.suburb and a.guide:
        try:
            print_report(check(a.suburb, a.postcode, a.type, a.land, a.guide))
        except ValueError as e:
            sys.exit(str(e))
    else:
        ap.error("give --suburb and --guide, or --csv")


if __name__ == "__main__":
    main()
