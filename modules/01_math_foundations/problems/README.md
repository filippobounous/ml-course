# Problem set — Week 1

Work on paper first, then typeset the cleanest version in `notebooks/`. Target: 6–8 hours.

## Theory (derivations)

1. **SVD identities.** For $A = U \Sigma V^\top$, prove: (a) $A^+ = V \Sigma^+ U^\top$; (b) $\|A\|_F^2 = \sum_i \sigma_i^2$; (c) the best rank-$k$ Frobenius approximation of $A$ is given by truncating the SVD (Eckart–Young).
2. **KL properties.** Show $D_{KL}(p\|q) \ge 0$ with equality iff $p=q$ a.e. Derive Pinsker's inequality $\|p-q\|_1 \le \sqrt{2 D_{KL}(p\|q)}$.
3. **Convexity.** Show that $f(x) = \log \sum_i e^{x_i}$ is convex. Derive its gradient and Hessian; identify when the Hessian is singular.
4. **GD convergence.** For $L$-smooth convex $f$, prove $f(x_T) - f^\star \le \tfrac{L\|x_0 - x^\star\|^2}{2T}$ under constant step $1/L$.

## Implementation (code)

5. Implement the **Moore–Penrose pseudoinverse** three ways in NumPy — via SVD, via normal equations with regularisation, and via `np.linalg.pinv`. Compare on a rank-deficient matrix and plot solution norm vs regularisation.
6. Implement **Gaussian maximum-likelihood** for a multivariate Gaussian. Compare to `numpy.cov`; verify bias correction.

## Applied

7. Simulate a **Langevin SDE** $dX_t = -\nabla U(X_t)\,dt + \sqrt{2}\,dW_t$ with $U(x) = \tfrac{x^4}{4} - \tfrac{x^2}{2}$. Plot the empirical stationary distribution vs the Gibbs form $\propto e^{-U}$.

## Grading

Auto-graded pieces (parts 5 and 6) live in `tests/week_01/`.
