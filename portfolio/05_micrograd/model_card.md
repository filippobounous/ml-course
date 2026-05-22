# Model card — Micrograd MLP (two-moons)

Following Mitchell et al. 2019. See the
[template](../model_card_template.md) for the full schema.

## Model details

- **Name.** Micrograd-style MLP for two-moons binary classification.
- **Architecture.** 2 hidden layers (tanh, then tanh), width 8 each;
  final linear → logit; cross-entropy. ~100 trainable scalar `Value`s.
- **Framework.** From-scratch scalar autograd (`mlcourse.autograd`),
  *not* PyTorch. The point of this artifact is the autograd engine.

## Intended use

- **Primary.** Pedagogical proof that reverse-mode autodiff +
  hand-written `Value` ops produce gradients matching `torch.autograd`
  bit-equivalently on small graphs.
- **Out-of-scope.** Any non-toy classification. The engine is
  scalar (no broadcasting, no vectorisation) — it cannot train any
  realistic model.

## Metrics

- Test accuracy on a 200-point held-out two-moons split.
- Slow-tier test asserts ≥ 88% after 20 epochs (`@pytest.mark.slow`).

## Evaluation / training data

- `sklearn.datasets.make_moons(n_samples=300, noise=0.20, random_state=0)`.
  Synthetic; no PII; no demographic factors.

## Quantitative analyses

| Metric | Value | Verified |
|---|---|---|
| Test acc (slow test) | ≥ 0.88 | ✅ (slow test enforces) |
| Gradient match vs torch on 6 toy graphs | bit-equivalent | ✅ (unit tests enforce) |

## Caveats

- The engine cannot scale past ~10³ ops without significant slowdown
  (Python-level scalar graph).
- Two-moons is *not* a meaningful benchmark — it's a sanity check.
  Don't quote 88% accuracy as evidence the engine is "good".
