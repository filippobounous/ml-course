# Hydra config tree

Hydra configs used by `mlcourse.Trainer` and the per-week training demos.

## Layout

```
configs/
├── trainer/
│   └── default.yaml      # base TrainerConfig (group default)
├── week06/
│   └── trainer_demo.yaml # toy regression demo
├── week07/
│   └── cifar10.yaml      # ResNet-18 + Grad-CAM + FGSM
└── week10/
    └── ddpm.yaml         # SmallUNet DDPM on FashionMNIST
```

## How per-week configs compose the trainer base

Every per-week config is `# @package _global_` so its keys land at the root
namespace (i.e. `cfg.trainer.lr` rather than `cfg.week07.trainer.lr`). They
pull in `trainer/default.yaml` as a group default and override fields:

```yaml
# @package _global_
defaults:
  - /trainer: default
  - _self_

trainer:
  max_epochs: 10
  lr: 0.01
  grad_clip_norm: 1.0
```

Override anything from the command line with the standard Hydra dot-syntax:

```bash
python portfolio/06_trainer/demo.py trainer.max_epochs=1
python portfolio/07_vision_classifier/demo.py quick=true
python portfolio/10_ddpm/train.py trainer.max_epochs=20 diffusion.T=500
```
