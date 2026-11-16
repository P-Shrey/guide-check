"""Measure the error the guide bands should be scaled by.

Why. The bands were scaled by the point model's typical error on held-out
calibration data (metro 14.3%, regional 16.0%). But the band is applied to the
COMPARABLE-SALES estimate for a property typed in by suburb, and that is a
noisier number: on the 160 controlled cases (src/controlled_cases.py) it
missed the sale price by a median 21%, so 13 of 40 guides set at the actual
sale price were flagged. The scale has to be the error of the thing the band
is applied to.

How. 100 sales per segment x property type, each typed in exactly as a user
would (suburb, postcode, type, land area, date), with the sale itself excluded
from its own comparables. The median absolute miss of the comparable-sales
estimate, per stratum, is written to artefacts/band_scale.json and serve.py
bands on it.

Which sales. We first measured on the calibration window (2025): metro 15.4%,
regional houses 15.3%, regional units 25.2%. That flatters the tool, because
those sales sit inside the comparable-sales index's own period, so their
comparables are contemporaneous. In real use the comparables are always older
than the guide (the index ends 2025-09-30), and on the test window the
comparable estimate missed by a median 21%. So the shipped scale is measured on
test-window sales (the conditions the tool actually runs in), EXCLUDING every
property used by src/controlled_cases.py, which therefore stays a clean
evaluation. Both measurements are kept in the JSON.

    python src/band_scale.py              test window, the shipped scale (~8 min)
    python src/band_scale.py --calib      calibration window, for comparison
"""
import json
import sys

import numpy as np
import pandas as pd

import config as C
import serve

PER_STRATUM, SEED = 100, 20261006
OUT = C.ART / "band_scale.json"


def main():
    df = pd.read_parquet(C.ROOT / "data" / "processed" / "sales_features.parquet",
                         columns=["suburb", "postcode", "property_type", "land_area_m2",
                                  "contract_date", "sale_price", "dist_cbd_km"])
    if "--calib" in sys.argv:
        window = "calibration"
        cal = df[(df["contract_date"] >= C.TRAIN_END) & (df["contract_date"] < C.CALIB_END)].copy()
    else:
        window = "test, excluding controlled cases"
        cal = df[df["contract_date"] >= C.CALIB_END].copy()
        used = pd.read_csv(C.REF / "controlled_cases.csv")
        used = set(zip(used["suburb"], used["sale_price"], used["contract_date"]))
        cal = cal[[(s, p, str(d)[:10]) not in used for s, p, d in
                   zip(cal["suburb"], cal["sale_price"], cal["contract_date"])]]
    cal["segment"] = np.where(cal["dist_cbd_km"] < 50, "metro", "regional")
    picks = cal.groupby(["segment", "property_type"]).sample(n=PER_STRATUM, random_state=SEED)

    errs = []
    for i, (key, s) in enumerate(picks.iterrows(), 1):
        r = serve.value_manual(suburb=s.suburb, postcode=str(s.postcode), property_type=s.property_type,
                               land_area_m2=None if pd.isna(s.land_area_m2) else s.land_area_m2,
                               when=s.contract_date, key=key)
        if r.get("comps_estimate"):
            errs.append({"segment": s.segment, "property_type": s.property_type,
                         "abs_err_pct": abs(r["comps_estimate"] / s.sale_price - 1) * 100})
        if i % 50 == 0:
            print(f"  {i}/{len(picks)}")

    e = pd.DataFrame(errs)
    scale = {f"{seg}|{pt}": round(float(g["abs_err_pct"].median()), 1)
             for (seg, pt), g in e.groupby(["segment", "property_type"])}
    scale["all"] = round(float(e["abs_err_pct"].median()), 1)
    OUT.write_text(json.dumps({"median_abs_error_pct": scale, "n": len(e),
                               "window": window, "seed": SEED,
                               "calibration_window_for_comparison": {
                                   "metro|house": 15.4, "metro|unit": 15.6, "regional|house": 15.3,
                                   "regional|unit": 25.2, "all": 17.5}}, indent=2))
    print(json.dumps(scale, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
