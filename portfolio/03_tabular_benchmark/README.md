# 03 — Tabular benchmark (logistic / RF / boosted trees)

Benchmark of classical tabular-ML methods on **UCI Adult (income > 50k)** with
honest calibration reporting.

## What's here

- `benchmark.py` — fetches UCI Adult via OpenML, fits five models, reports ROC
  AUC / PR AUC / Brier score / fit time, and saves ROC, PR, and calibration
  panels to `roc_pr_calibration.png`. Also writes a small `results.md` table.
- Models: logistic regression, random forest, scikit-learn gradient boosting;
  XGBoost and LightGBM are added when they are installed (see `.[ops]`).

## Reproduce

```bash
python -m pip install -e ".[dev,ops]"   # installs xgboost + lightgbm
python portfolio/03_tabular_benchmark/benchmark.py
```

Runs in ~1 minute on a single CPU core; network access is needed only on the
first run for the OpenML fetch (cached afterwards).

## Tests

`tests/week_03/` covers the IRLS logistic regression solver against
scikit-learn on a synthetic Bernoulli problem plus the information-gain
helpers.

## Findings template

Fill in after running (`results.md` is auto-generated):

1. Which tree ensemble wins, and by how much?
2. Is logistic regression competitive on AUC but worse on PR? (Hint: yes, due
   to class imbalance.)
3. Which model is best-calibrated out of the box? (Hint: typically logistic
   regression and scikit-learn GBDT; XGBoost / LightGBM often over-confident
   and benefit from Platt or isotonic rescaling.)
