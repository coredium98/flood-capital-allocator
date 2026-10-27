"""Matplotlib chart builders for threshold cost-benefit and classification trade-offs."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def plot_net_benefit_and_roi(results_df: pd.DataFrame, optimal_threshold: float) -> plt.Figure:
    """Render side-by-side Net Benefit and ROI curves across probability thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        results_df["threshold"],
        results_df["net_benefit"],
        linewidth=2.5,
        color="#2ecc71",
        marker="o",
        markersize=4,
    )
    axes[0].axvline(
        optimal_threshold,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Optimal: {optimal_threshold:.2f}",
    )
    axes[0].axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.3)
    axes[0].set_xlabel("Risk Threshold", fontsize=11)
    axes[0].set_ylabel("Net Benefit ($M)", fontsize=11)
    axes[0].set_title("Net Benefit by Probability Threshold", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        results_df["threshold"],
        results_df["roi"],
        linewidth=2.5,
        color="#3498db",
        marker="o",
        markersize=4,
    )
    axes[1].axvline(
        optimal_threshold,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Optimal: {optimal_threshold:.2f}",
    )
    axes[1].axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.3)
    axes[1].set_xlabel("Risk Threshold", fontsize=11)
    axes[1].set_ylabel("ROI (%)", fontsize=11)
    axes[1].set_title("Return on Investment by Threshold", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_cost_vs_prevented_damage(results_df: pd.DataFrame, optimal_threshold: float) -> plt.Figure:
    """Render total intervention cost against prevented flood damage across thresholds."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        results_df["threshold"],
        results_df["intervention_cost"],
        linewidth=2.5,
        color="#e74c3c",
        marker="o",
        markersize=4,
        label="Intervention Cost",
    )
    ax.plot(
        results_df["threshold"],
        results_df["prevented_damage"],
        linewidth=2.5,
        color="#2ecc71",
        marker="s",
        markersize=4,
        label="Prevented Damage",
    )
    ax.axvline(
        optimal_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Optimal Threshold: {optimal_threshold:.2f}",
    )
    ax.set_xlabel("Risk Threshold", fontsize=11)
    ax.set_ylabel("Amount ($M)", fontsize=11)
    ax.set_title("Intervention Cost vs. Prevented Damage", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_precision_recall_and_volume(
    results_df: pd.DataFrame, optimal_threshold: float
) -> plt.Figure:
    """Render Precision/Recall trade-off and total flagged areas across thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        results_df["threshold"],
        results_df["precision"],
        linewidth=2.5,
        color="#9b59b6",
        marker="o",
        markersize=4,
        label="Precision",
    )
    axes[0].plot(
        results_df["threshold"],
        results_df["recall"],
        linewidth=2.5,
        color="#f39c12",
        marker="s",
        markersize=4,
        label="Recall",
    )
    axes[0].axvline(
        optimal_threshold,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Optimal: {optimal_threshold:.2f}",
    )
    axes[0].set_xlabel("Risk Threshold", fontsize=11)
    axes[0].set_ylabel("Score", fontsize=11)
    axes[0].set_title("Precision and Recall by Threshold", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        results_df["threshold"],
        results_df["areas_flagged"],
        linewidth=2.5,
        color="#1abc9c",
        marker="o",
        markersize=4,
    )
    axes[1].axvline(
        optimal_threshold,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Optimal: {optimal_threshold:.2f}",
    )
    axes[1].set_xlabel("Risk Threshold", fontsize=11)
    axes[1].set_ylabel("Number of Areas", fontsize=11)
    axes[1].set_title("Areas Flagged for Intervention", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
