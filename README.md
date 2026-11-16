# Guide Check

**Is this advertised NSW price guide defensible against comparable sales?**

Guide Check takes a suburb, a property type, a land area and an advertised price guide, and
tells a buyer where that guide sits among the 25 most comparable recent sales — adjusted to
the subject property by a LightGBM model trained on 1.25 million NSW Valuer General sales.
Every answer ships with its evidence (the actual sales, with addresses and prices) and its
measured limits (how often the model is actually right).

**Live demo: https://guide-check.vercel.app**

---

## The problem

In NSW a buyer sees a guide of $1.1M, pays for a building and pest report, a strata report
and a solicitor's contract review, takes time off for inspections — and the property sells
for $1.4M. They were never in the running.

This is measurable. A single-Saturday audit of 287 NSW properties found half sold at least
10% above their guide; Victoria's underquoting taskforce has logged more than 8,800
complaints and issued more than 260 fines.

The usual tools answer the wrong question: *"Did it sell for more than the guide?"* A hot
auction pushes prices past any honest estimate, and that question can only be answered after
the money is spent. NSW law asks something narrower that **can** be answered before the
auction: *is the guide a reasonable estimate given comparable sales?* Comparable sales are
public. Guide Check asks that question.

## Business value

| Who | What they get |
|---|---|
| **Buyers** | A ten-second sanity check before spending $500–$2,000 and a fortnight on a property they may be priced out of. |
| **Buyer's agents / advocates** | A repeatable, evidence-backed first pass on every listing a client sends, with the comparables already pulled. |
| **Mortgage brokers / lenders** | An early signal that a client's budget and a target property are mismatched, before pre-approval work is wasted. |
| **Regulators / consumer groups** | A screening tool that ranks listings by how far below the comparable evidence their guides sit, to prioritise review. |
| **Selling agents** | A check that their own guide is defensible against the comparables a buyer will see. |

The output is deliberately a **statistical observation, not an accusation**: whether an agent
underquoted depends on their own reasonable estimate at listing time, which no model can see.
That framing is what makes the tool usable in a consumer product without legal exposure.

## How it works

```
NSW Valuer General PSI (1.5M rows)
        │  clean.py      residential only, 2019+, price bounds, de-dup, nominal-sale filter
        ▼
1,249,163 clean sales ── features.py ── structural · spatial · temporal · suburb index
        │  split.py      frozen temporal split (test = sales from 2025-10-01),
        │                CV grouped by ABS SA2 so no area leaks between folds
        ▼
LightGBM point model ──► conformalised quantile interval (CQR, drift-corrected)
        │
        ▼
Subject property ──► 25 nearest recent comparables (same type, nearby, similar land)
                     each adjusted by the model's price ratio subject/comparable
                     ──► adjusted distribution ──► guide percentile + gap in typical errors
                     ──► band: consistent / low end / below / well below
```

1. **Comparable sales.** A `NearestNeighbors` index over recent sales (2-year lookback)
   returns the 25 closest of the same property type by location and land size.
2. **Adjustment.** No two properties are identical, so each comparable's sale price is
   scaled by the model's predicted ratio between the subject and that comparable.
3. **Assessment.** The guide is located in the adjusted distribution as a percentile, and
   its gap to the comparable estimate is expressed in units of *that segment's measured
   typical error* (metro vs regional, house vs unit). Bands fire at 0.5×, 1× and 2× typical
   error; with fewer than 8 comparables no band is given.
4. **Explanation.** SHAP drivers, the comparables themselves, the prediction interval and
   its measured (not nominal) coverage are all returned with every result.

### Why the bands are scaled by error, not percentile

An earlier version flagged guides below the 5th/15th percentile of comparables. Because the
adjusted comparables spread widely (IQR ≈ 30% of the median), a real 10–15% underquote — the
size seen in NSW enforcement cases — landed near the 30th percentile and was never flagged.
Scaling the gap by measured error makes the statement defensible: *"more than one typical
error below the comparable evidence is not explainable by the model's own noise."*

## Results

All numbers are on a test set of **119,095 sales** frozen before any modelling began
(`results/results-20260927-212806.md`).

| Metric | Value |
|---|---|
| Point accuracy (MAPE) | **25.81%** vs 41.33% suburb-median baseline (**37.5% better**) and 36.57% linear regression |
| Median absolute % error | 16.8% |
| Interval coverage | nominal 80%, **measured 77.6%** — the measured figure is what the app shows users |
| Most valuable feature group | temporal (+4.77 pp MAPE when removed), then structural (+2.71), spatial (+1.01) |
| LightGBM vs XGBoost | 25.81% vs 25.89% MAPE — LightGBM kept |

**Guide assessment, 160 labelled cases** (40 held-out sales × guides at 70/80/90/100% of the
actual sale price, `results/controlled_cases.md`):

| | |
|---|---|
| Underquoted guides flagged (recall) | 56% (45/80) — 65% at a 30% underquote |
| Defensible guides wrongly flagged | 25% (10/40) |
| Precision | 82% |

The harness itself is validated: `src/make_fake_predictions.py` generates predictions with a
known +12% bias and ~70% interval coverage, and the harness must recover those numbers before any real
model is scored.

## Tech stack

- **Modelling:** LightGBM, scikit-learn (NearestNeighbors, GroupKFold), conformalised quantile
  regression, SHAP, pandas / NumPy / PyArrow
- **Backend:** Flask — five static pages plus a JSON API
- **Frontend:** plain HTML/CSS/JS, no framework or build step; hand-built SVG charts
  (see `docs/DESIGN-NOTES.md`)
- **Hosting:** Vercel Python serverless function (`vercel.json`, Python 3.12)

## Run it locally

```bash
pip install -r requirements.txt     # Python 3.11 or 3.12; versions are pinned
python app.py                       # open http://127.0.0.1:5000
```

The trained models (`artefacts/`) and the feature table and split (`data/processed/`) are in
the repo via **Git LFS**, so no download or retraining is needed — run `git lfs install`
before cloning. On Linux, LightGBM also needs OpenMP (`apt install libgomp1`).

First load takes 30–45 s (1.25M-row feature table plus the SHAP explainer) and about 1.2 GB
of memory; every check after that takes about a second.

### Command line

```bash
python check.py --suburb PARRAMATTA --postcode 2150 --type house --land 450 --guide 950000
python check.py --csv examples/demo-inputs.csv --out results.csv   # batch-score a CSV
python selftest.py                  # demo cases band correctly, banding is monotone
python src/harness.py               # rescore the frozen test set (~1 min)
python src/controlled_cases.py      # rerun the 160 labelled guide cases (~3 min)
python src/run_all.py               # rebuild everything from raw data (15–25 min)
```

`run_all.py` downloads the raw PSI extract (~74 MB zipped) into `data/raw/` on first run.

### API

```http
POST /api/check
Content-Type: application/json

{"suburb": "PARRAMATTA", "postcode": "2150", "property_type": "house",
 "land_area_m2": 450, "guide": 950000}
```

Returns the band, percentile, gap in typical errors, comparable estimate, the 25 adjusted
comparables, point estimate and interval, SHAP drivers and the measured-coverage note.
`GET /api/suburbs` returns the suburb list for autocomplete; `GET /healthz` is a liveness probe.

### Deploy

```bash
vercel --prod
```

`.vercelignore` keeps evaluation files out of the function bundle. If deploying from GitHub
instead of the CLI, enable **Git LFS** in the Vercel project settings so the model files are
pulled.

## Data

- **NSW Valuer General Property Sales Information** (bulk PSI), via the
  [nswpropertysalesdata.com](https://nswpropertysalesdata.com) CSV mirror. 1,508,054 rows,
  effectively 2019 → Aug 2026; 1,249,163 after cleaning. Every cleaning decision and row count
  is in `docs/DATA-REPORT.md` and `src/config.py`.
- **Australian postcodes** ([matthewproctor/australianpostcodes](https://github.com/matthewproctor/australianpostcodes))
  for suburb centroids and SA2 codes.

PSI has no bedrooms, bathrooms or coordinates, so the model works at suburb level.

## Project structure

```
app.py                  Flask app: five pages, /api/suburbs, POST /api/check, /healthz
check.py                the tool without the front end: one property or a CSV
selftest.py             one-command regression check
static/                 site pages (home, how to use, the tool, how it works, limits, 404)
static/assets/          app.css, app.js (shared chrome), check.js (tool + charts)
src/config.py           every tunable and cleaning decision in one place
src/clean.py split.py   raw PSI → clean sales → frozen train/calib/test split
src/features.py         feature engineering (fitted on train only, no leakage)
src/baselines.py        suburb-median and linear-regression baselines
src/train_point.py      LightGBM vs XGBoost point model
src/train_interval.py   quantile + conformal intervals
src/improve_v2.py       drift-corrected conformal interval (v2)
src/comps*.py           comparable-sales index and the comps-anchored estimate
src/serve.py            scoring, banding and the guide assessment
src/band_scale.py       measures the comps estimate's error the bands are scaled by
src/harness.py          evaluation harness
src/ablation.py         feature-group ablations
src/controlled_cases.py 160 labelled guide cases built from held-out sales
src/hardcases.py        scoring for real labelled listings (template in data/reference/)
src/make_fake_predictions.py   proves the harness measures error correctly
src/run_all.py          rebuild the whole pipeline in dependency order
artefacts/              trained models and indexes (Git LFS)
data/processed/         feature table and frozen split (Git LFS)
data/reference/         labelled cases and templates
examples/               demo inputs and their expected outputs
predictions/ results/   test-set predictions (gzipped) and evaluation reports
docs/                   data report and front-end design notes
```

## Limitations

- **Suburb-level, not street-level.** Two properties of the same type and land size in one
  suburb get the same estimate; condition, views, renovations and bedrooms are invisible.
- **Interval slightly narrow:** 77.6% measured against 80% nominal, and the app says so.
- **Moderate recall:** about half of 20–30% underquotes are flagged, and one in four guides
  set at the true price gets a flag. Weakest on regional units, where comparables are thin.
- **Evidence ages.** Comparables come from a 2-year lookback; fast-moving markets drift.
- **Not a valuation, not legal or financial advice,** and not an allegation of underquoting.

## Contributors

Shrey Patel, Sushant Shrestha, Dheeransh Jain.

## License

Code is released under the [MIT License](LICENSE). The underlying sales data remains subject to
the NSW Valuer General's terms of use.
