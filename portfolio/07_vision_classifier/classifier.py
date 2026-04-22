"""CIFAR-10 vision classifier — ResNet-18 from scratch + transfer-learning baseline.

Torch is required (`pip install -e '.[dl,ops]'`). The module imports torch at
the top so `python -m classifier` gives a clean error if torch is missing;
the pytest suite under `tests/week_07/` gates everything on
`pytest.importorskip("torch")`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError as e:  # pragma: no cover - environment guard
    raise ImportError(
        "This module requires PyTorch. Install with `pip install -e '.[dl,ops]'`."
    ) from e


HERE = Path(__file__).resolve().parent

# CIFAR-10 normalisation (standard).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# -----------------------------------------------------------------------------
# Data


def get_cifar10_loaders(
    data_root: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
    augment: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 (first call) and return train/test DataLoaders."""
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_tf = (
        transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        if augment
        else transforms.Compose([transforms.ToTensor(), normalize])
    )
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])
    train = CIFAR10(data_root, train=True, download=True, transform=train_tf)
    test = CIFAR10(data_root, train=False, download=True, transform=test_tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


# -----------------------------------------------------------------------------
# Model factory


def resnet18_for_cifar(num_classes: int = 10) -> nn.Module:
    """`torchvision.models.resnet18` adapted for 32×32 CIFAR input.

    Replaces the 7×7 stride-2 stem with a 3×3 stride-1 conv and drops the
    initial maxpool — the standard CIFAR recipe from He et al. (2015).
    """
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def transfer_resnet18(num_classes: int = 10, freeze_backbone: bool = True) -> nn.Module:
    """Pretrained ResNet-18 backbone with a fresh linear head.

    Uses the torchvision default weights (DEFAULT -> IMAGENET1K_V1 today).
    """
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# -----------------------------------------------------------------------------
# Training & evaluation


@dataclass
class EpochLog:
    train_loss: float
    test_loss: float
    test_acc: float


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss)
            correct += int((logits.argmax(1) == y).sum())
            total += y.size(0)
    return total_loss / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * y.size(0)
        total += y.size(0)
    return total_loss / total


# -----------------------------------------------------------------------------
# Grad-CAM


class GradCAM:
    """Minimal Grad-CAM implementation against a single target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._h1 = target_layer.register_forward_hook(self._save_activations)
        self._h2 = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _m, _i, out: torch.Tensor) -> None:
        self.activations = out.detach()

    def _save_gradients(self, _m, _gin, gout: tuple[torch.Tensor, ...]) -> None:
        self.gradients = gout[0].detach()

    def close(self) -> None:
        self._h1.remove()
        self._h2.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __call__(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        """Return a (H, W) heatmap for the first sample in `x`."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if target_class is None:
            target_class = int(logits.argmax(1)[0])
        self.model.zero_grad(set_to_none=True)
        logits[0, target_class].backward()
        assert self.activations is not None and self.gradients is not None
        weights = self.gradients.mean(dim=(2, 3))  # (B, C)
        cam = (weights[0, :, None, None] * self.activations[0]).sum(dim=0)
        cam = torch.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


# -----------------------------------------------------------------------------
# FGSM adversarial example


def fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Fast Gradient Sign Method (Goodfellow et al. 2014)."""
    model.eval()
    x = x.clone().requires_grad_(True)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert x.grad is not None
    return (x + epsilon * x.grad.sign()).detach()
