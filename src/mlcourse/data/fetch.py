"""CLI entry point for `make fetch-data`.

Populated incrementally as each week's datasets are introduced. Until then this
prints a manifest of what each week expects so learners know where to look.
"""

from __future__ import annotations

MANIFEST: dict[str, list[str]] = {
    "week_03": ["UCI Adult", "Covertype"],
    "week_04": ["Ken French industry portfolios", "Old Faithful"],
    "week_07": ["CIFAR-10 (torchvision)"],
    "week_08": ["TinyStories", "Tiny Shakespeare"],
    "week_09": ["Alpaca-cleaned", "UltraFeedback (DPO)"],
    "week_10": ["FashionMNIST (torchvision)"],
    "week_11": ["gymnasium: CartPole, LunarLander, Pendulum"],
    "week_12": ["Fama-French factor CSVs (via yfinance / statsmodels)"],
}


def main() -> None:
    print("Dataset manifest (expected data sources per week):")
    for week, datasets in MANIFEST.items():
        print(f"  {week}: {', '.join(datasets)}")
    print(
        "\nFetcher stubs land as each week's module is authored. "
        "For now, follow the per-module README for dataset download instructions."
    )


if __name__ == "__main__":
    main()
