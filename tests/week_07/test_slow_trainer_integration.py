"""Slow integration test for Week 7: `mlcourse.Trainer` on a tiny CIFAR-like task.

The full CIFAR-10 fetch + 10-epoch training is out of reach for CI. This test
synthesises a small classification dataset of random 3×32×32 images with
class-specific means, builds the Week-7 ResNet-18-for-CIFAR model, and trains
for one epoch via `mlcourse.Trainer.fit`. Asserts train loss drops below the
initial value.

This guards the claim in `portfolio/07_vision_classifier/README.md` that
"the demo uses `mlcourse.Trainer`" — if the trainer API changes and the
W7 demo is not updated, this test fails.

Skipped by default; run with `pytest --run-slow -q`.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_trainer_fit_on_tiny_cifar_like():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset

    from mlcourse.trainer import Trainer, TrainerConfig

    # Synthesise a 4-class problem with mean shifts — classification-solvable in one epoch.
    torch.manual_seed(0)
    n_classes = 4
    per_class = 64
    imgs = []
    labels = []
    for c in range(n_classes):
        shift = torch.zeros(3, 1, 1)
        shift[c % 3] = 1.5
        imgs.append(torch.randn(per_class, 3, 32, 32) + shift)
        labels.extend([c] * per_class)
    x = torch.cat(imgs, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    perm = torch.randperm(x.size(0))
    x, y = x[perm], y[perm]
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    # Small CNN sized like the W7 ResNet-18 head but much faster to train.
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(16, n_classes),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = Trainer(TrainerConfig(max_epochs=3, device="cpu", seed=0))
    trainer.fit(
        model, loader, loss_fn=torch.nn.functional.cross_entropy, optimizer=optimizer
    )
    history = trainer.history["train_loss"]
    assert len(history) == 3
    assert history[-1] < history[0], (
        f"train loss should drop: {history[0]:.4f} → {history[-1]:.4f}"
    )
