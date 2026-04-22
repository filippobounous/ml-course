# Problem set — Week 4

## Theory

1. **PCA as MLE.** Derive probabilistic PCA. Show that in the zero-noise limit its MLE recovers standard PCA.
2. **EM for GMM.** Derive the E-step and M-step updates. Prove monotone likelihood improvement (or explain the Jensen step).
3. **k-means convergence.** Show Lloyd's algorithm decreases the within-cluster-sum-of-squares at every step.

## Implementation

4. Implement **GMM-EM** in NumPy on Old Faithful; compare to `sklearn.mixture.GaussianMixture`. Plot log-likelihood vs iteration.
5. Implement **PCA via SVD** and via power iteration; compare stability and runtime.

## Applied (portfolio artifact)

6. **PCA stat-arb on Ken French 49 industry portfolios.** Construct residual portfolios as in Avellaneda–Lee; compute in-sample and out-of-sample Sharpe; add transaction costs; report with walk-forward splits. Deliver in `portfolio/04_pca_statarb/`.

## Grading

Tests in `tests/week_04/` check GMM-EM log-likelihoods improve monotonically and PCA eigenvectors match sklearn to sign.
