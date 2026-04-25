# Capstone proposal

*Due end of Week 8. One page maximum.*

## Goal

*One sentence. What will you have at the end?*

> Example: "Train a PINN to solve the 1-D Burgers' equation and reproduce
> Raissi et al. 2019 Figure 2 to within 10× of the paper's error on MPS."

## Track

- [ ] A — physics / scientific ML
- [ ] B — quantitative finance

## Dataset / environment

- **Source.** <URL + licence>
- **Size.** <rows / GB / number of trajectories>
- **Notes.** <any preprocessing that matters>

## Primary metric

*One number. If you cannot state it now, the capstone is not yet scoped.*

## Baselines

- **Trivial baseline.** <e.g. zero predictor / buy-and-hold / analytical
  solution>
- **Strong baseline.** <classical method that a thoughtful practitioner
  would ship>
- **My contribution.** <what the capstone adds on top>

## Compute budget

- **Total wall-clock.** <hours on MPS / CPU>
- **Cap.** <hard limit before I cut scope>

## Timeline

| Week | Milestone |
|---|---|
| 8 | Proposal + tiny-data end-to-end pipeline (this file + a `minimal.ipynb`) |
| 9 | First full-scale training run; find out what breaks |
| 10 | Fix what broke; add at least one ablation |
| 11 | Reproducibility pass: seed, Hydra config, model/dataset card |
| 12 | Ship. Move artifact to `portfolio/12_capstone/` and write findings |

## Risks

- <concrete failure mode 1 + mitigation>
- <concrete failure mode 2 + mitigation>

## Stretch goals

- <nice-to-have>
- <nice-to-have>
