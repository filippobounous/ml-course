# Paper reproduction plan

*One-page plan; fill in before coding.*

## Target

- **Paper.** <author, year, title — one of: LoRA (Hu 2021), DDPM (Ho 2020),
  PPO (Schulman 2017), PINNs (Raissi 2019).>
- **Specific figure / table to reproduce.** <exact reference>

## Data

- **Dataset.** <name + URL + size in GB>
- **Preprocessing.** <one paragraph>

## Setup

- **Model.** <size; if rescaling from the paper, state ratio>
- **Base checkpoint.** <if applicable>
- **Hyperparameters.** <state the subset that will be fixed at paper values;
  list the one or two that we will ablate>

## Ablation (minimum one extra config)

- **Variable.** <which hyperparameter>
- **Values.** <small set; 2–4 points is plenty>
- **Hypothesis.** <what we expect to see and why>

## Compute budget

- **Target total compute.** <X GPU-hours / Y wall-clock hours on MPS / CPU>
- **Cap.** <maximum before abandoning — e.g. 10 hours>

## Success criteria

- **Qualitative match.** <monotonicity / shape of the curve>
- **Quantitative match.** <final metric within X% of paper>
- **Deliverable.** `<figure_name>.png` + a short `findings.md` with the
  ablation table and a one-paragraph "what I saw that surprised me".

## Risks

- <one or two concrete failure modes; e.g. "DDIM sampler is slow at 1000 steps
   on MPS — mitigate by running on a small val subset.">
