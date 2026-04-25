# Problem set — Week 5

## Theory

1. **Backprop by hand.** Derive gradients for a 2-layer MLP with tanh + softmax + cross-entropy on a batch of inputs. State the backward-pass equations explicitly.
2. **Glorot init.** Show that with linear activations and $W_{ij} \sim \mathcal{N}(0, 1/n_\text{in})$ (or the symmetric variant) the per-layer activation variance is preserved.
3. **Adam.** Derive the bias-corrected first- and second-moment estimates. Explain why bias correction matters in the first ~1/(1−β) steps.

## Implementation (portfolio)

4. Build the **micrograd-style autograd engine** in `portfolio/05_micrograd/`:
   - scalar `Value` class with `+ - * / ** exp tanh relu` and `backward()`
   - MLP, Layer, Neuron abstractions
   - ≥6 unit tests comparing to torch-autograd or analytical gradients
   - demo notebook training on the two-moons dataset

## Applied

5. **Init vs optimiser ablation.** On MNIST with a 2-hidden-layer MLP (use sklearn-loader + your autograd or torch), ablate over {Glorot, He} × {SGD, Adam} × {BN on/off}. Report final test accuracy and training curves. Short write-up: which combinations struggle, and why?

## Grading

Tests in `tests/week_05/` compare your engine's gradients to numerical gradients on a handful of inputs.
