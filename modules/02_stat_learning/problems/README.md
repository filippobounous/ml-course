# Problem set — Week 2

## Theory

1. **Bias–variance decomposition.** Derive for squared loss. Show explicitly the decomposition for ridge regression and plot bias/variance/total error vs $\lambda$ on a toy problem.
2. **Ridge ≡ MAP.** Show ridge regression coincides with MAP under a Gaussian prior. State the prior's variance as a function of $\lambda$ and noise $\sigma^2$.
3. **Closed form for OLS.** Derive $\hat{\beta} = (X^\top X)^{-1} X^\top y$ from first-order conditions. Show $\mathbb{E}[\hat{\beta}] = \beta$ and $\operatorname{Var}(\hat{\beta}) = \sigma^2 (X^\top X)^{-1}$.
4. **CV bias.** Prove leave-one-out CV is an approximately unbiased estimator of the generalisation error.

## Implementation

5. **NumPy linear-models library** (portfolio artifact): closed-form OLS + SGD + ridge + lasso (coordinate descent) + K-fold CV + unit tests.
6. Reproduce a **regularisation path** figure (ridge and lasso) on a simulated problem with $p \gg n$.

## Applied

7. Formalise a simple **MDP** (e.g. grid world) and compute the optimal state-value function via policy iteration (tabular). This is preparation for Week 11.

## Grading

Tests in `tests/week_02/` check the library's API (fit/predict shapes, CV scores monotonicity in $\lambda$ for ridge on a known problem).
