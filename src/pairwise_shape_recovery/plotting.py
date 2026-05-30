"""Matplotlib plotting routines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_correlation_heatmap(
    correlation: pd.DataFrame, output_path: str | Path
) -> None:
    """Save a correlation heatmap with values annotated in each cell."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(correlation.columns)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(
        correlation.to_numpy(dtype=float),
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    values = correlation.to_numpy(dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    ax.set_xlabel("Statistic")
    ax.set_ylabel("Statistic")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
