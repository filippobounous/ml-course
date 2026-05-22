# Model card template

Following Mitchell et al. 2019, [*Model Cards for Model Reporting*](https://arxiv.org/abs/1810.03677).

Each artifact under `portfolio/` that produces a trained model carries a
`model_card.md` filled from this template. The point of a model card is
not the model — it's making the model's **intended use, evaluation
context, and known limitations** visible to anyone who didn't train it.

Sections marked **Verified ⏳** are filled in after a real-hardware run
ticks the row in the [verified-vs-aspirational table](../README.md#verified-vs-aspirational-honesty-table).

---

## Model details

- **Name.** `<artifact-name>` — e.g. "Micrograd MLP for two-moons".
- **Version.** `<git-sha>` of the training run.
- **Date.** `<YYYY-MM-DD>`.
- **Architecture.** One-paragraph summary: layer types, parameter count,
  input shape, output shape.
- **License.** Inherits the repo's LICENSE.
- **Citation.** If reproducing a paper, cite it. Otherwise:
  *"`<week-N>` artifact, mlcourse"*.
- **Contact.** Repo owner (see top-level README).

## Intended use

- **Primary use case.** One sentence: what is this model for, in this
  course context?
- **Primary intended users.** *Course learners + their employers /
  collaborators reading the portfolio.*
- **Out-of-scope.** What this model should *not* be used for —
  production deployment without re-training, sensitive decisions, etc.

## Factors

- **Relevant factors.** Demographic / domain factors that could affect
  performance. (For most W-N course artifacts the data is synthetic /
  toy, so this is "n/a — synthetic data.")
- **Evaluation factors.** Which factors were actually held constant
  vs varied during evaluation (e.g. random seed, batch size, dataset
  split).

## Metrics

- **Model performance measures.** Headline metric + any secondary
  metrics. Specify units.
- **Decision thresholds.** If this is a classifier, what cutoff is
  used and why.
- **Variation approaches.** How are results aggregated across seeds /
  folds? (e.g. "mean ± std over 4 seeds".)

## Evaluation data

- **Datasets.** Name + version + size + URL.
- **Motivation.** Why these datasets — what aspect of the model are
  they testing?
- **Preprocessing.** Normalisation, splits, augmentation.

## Training data

- **Datasets.** Name + version + size + URL. (Often the same as
  evaluation; some artifacts use a held-out split.)
- **Motivation.** Why this distribution.
- **Preprocessing.** Same as eval, plus train-only augmentation
  (e.g. random crop / flip for vision).

## Quantitative analyses

- **Unitary results.** Headline number + 95% CI or std.
  **Verified ⏳ / ✅** — flip to ✅ when a real-hardware run lands.
- **Intersectional results.** For classifiers, per-class /
  per-subgroup breakdown. (Often "n/a" for the course artifacts.)

## Ethical considerations

- **Sensitive use cases.** None of the W-N artifacts are intended for
  high-stakes decisions. State this explicitly so it's not assumed.
- **Mitigation.** Per-artifact: explicit out-of-scope statements,
  randomised seeds reported, no PII in training data.

## Caveats and recommendations

- **Known failure modes.** Each artifact has at least one. Be honest.
- **Confidence interval.** Single-seed runs have very wide CIs on
  most metrics; report std across ≥ 3 seeds before claiming a number.
- **Reproducibility.** The artifact's `make reproduce` (or
  `python demo.py`) should hit the headline metric ± a single std.
