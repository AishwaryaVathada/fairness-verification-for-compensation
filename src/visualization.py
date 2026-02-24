"""
visualization.py
----------------
Publication-quality visualization functions for compensation analysis.

All functions return matplotlib Figure objects and optionally save to disk.
Uses a consistent style across the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import config as cfg

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
STYLE_PARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
}

def apply_style():
    """Apply project-wide matplotlib style."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update(STYLE_PARAMS)

apply_style()


def _save(fig: plt.Figure, filename: str | None, directory: Path = cfg.FIGURES_DIR):
    """Save figure if filename is provided."""
    if filename:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / filename, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Categorical frequency plots
# ---------------------------------------------------------------------------

def plot_categorical_distributions(
    df: pd.DataFrame,
    cols: Sequence[str],
    save_prefix: str | None = None,
) -> list[plt.Figure]:
    """Bar charts of value counts for categorical columns."""
    figs = []
    for col in cols:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        vc = df[col].value_counts()
        vc.plot.bar(ax=ax, color=sns.color_palette("muted", len(vc)), edgecolor="white")
        ax.set_title(f"Distribution of {col}")
        ax.set_ylabel("Count")
        ax.set_xlabel(col)
        ax.tick_params(axis="x", rotation=45)
        for i, v in enumerate(vc):
            ax.text(i, v + 0.5, str(v), ha="center", fontsize=8)
        fig.tight_layout()
        _save(fig, f"{save_prefix}_{col.lower().replace(' ', '_')}.png" if save_prefix else None)
        figs.append(fig)
    return figs


# ---------------------------------------------------------------------------
# Numeric distribution plots
# ---------------------------------------------------------------------------

def plot_numeric_distributions(
    df: pd.DataFrame,
    cols: Sequence[str],
    save_prefix: str | None = None,
) -> list[plt.Figure]:
    """Histogram + KDE + rug plot for numeric columns."""
    figs = []
    for col in cols:
        if col not in df.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram + KDE
        ax = axes[0]
        data = df[col].dropna()
        ax.hist(data, bins=40, density=True, alpha=0.6, color="steelblue", edgecolor="white")
        xs = np.linspace(data.min(), data.max(), 300)
        kde = stats.gaussian_kde(data)
        ax.plot(xs, kde(xs), color="darkred", lw=2)
        ax.set_title(f"{col} - Distribution")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")

        # QQ plot
        ax2 = axes[1]
        stats.probplot(data, dist="norm", plot=ax2)
        ax2.set_title(f"{col} - Q-Q Plot")
        ax2.get_lines()[0].set_markerfacecolor("steelblue")
        ax2.get_lines()[0].set_markersize(3)

        fig.tight_layout()
        _save(fig, f"{save_prefix}_{col.lower().replace(' ', '_')}.png" if save_prefix else None)
        figs.append(fig)
    return figs


# ---------------------------------------------------------------------------
# Box / Violin plots by group
# ---------------------------------------------------------------------------

def plot_boxplot_by_group(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    order: Sequence[str] | None = None,
    title: str | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """Boxplot of a numeric variable across groups, ordered."""
    if order is None:
        order = df.groupby(group_col)[value_col].median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        x=group_col, y=value_col, data=df, order=order,
        hue=group_col, palette="Set2", dodge=False, legend=False,
        ax=ax, flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    ax.set_title(title or f"{value_col} by {group_col}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_violin_by_group_and_hue(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    hue_col: str,
    order: Sequence[str] | None = None,
    title: str | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """Split violin plot for comparing subgroups."""
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.violinplot(
        x=group_col, y=value_col, hue=hue_col, data=df,
        order=order, split=True, inner="quart", palette="viridis",
        ax=ax,
    )
    ax.set_title(title or f"{value_col} by {group_col} and {hue_col}")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title=hue_col, loc="upper left")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# Confidence interval plots
# ---------------------------------------------------------------------------

def plot_grouped_ci(
    ci_df: pd.DataFrame,
    group_col: str,
    sub_group_col: str | None = None,
    value_label: str = "Value",
    title: str | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """
    Plot confidence intervals from a DataFrame produced by
    ``inferential.grouped_ci()``.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    if sub_group_col and sub_group_col in ci_df.columns:
        sub_groups = ci_df[sub_group_col].unique()
        offsets = np.linspace(-0.2, 0.2, len(sub_groups))
        x_labels = ci_df[group_col].unique()
        x_positions = np.arange(len(x_labels))

        for sg, offset in zip(sub_groups, offsets):
            subset = ci_df[ci_df[sub_group_col] == sg]
            # Align to x_positions
            pos_map = {label: i for i, label in enumerate(x_labels)}
            xs = [pos_map[g] + offset for g in subset[group_col]]
            ax.errorbar(
                xs, subset["point_estimate"],
                yerr=subset["point_estimate"] - subset["ci_low"],
                fmt="o", capsize=5, markersize=7, label=sg,
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.legend(title=sub_group_col)
    else:
        x = ci_df[group_col]
        ax.errorbar(
            x, ci_df["point_estimate"],
            yerr=ci_df["point_estimate"] - ci_df["ci_low"],
            fmt="o-", capsize=5, markersize=7, color="steelblue",
        )
        ax.tick_params(axis="x", rotation=45)

    ax.set_ylabel(value_label)
    ax.set_title(title or f"Confidence Intervals by {group_col}")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    save_as: str | None = None,
) -> plt.Figure:
    """Annotated heatmap of a correlation matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# Bootstrap distribution plot
# ---------------------------------------------------------------------------

def plot_bootstrap_distribution(
    boot_result: dict,
    title: str = "Bootstrap Sampling Distribution",
    save_as: str | None = None,
) -> plt.Figure:
    """Histogram of bootstrap estimates with CI lines."""
    fig, ax = plt.subplots(figsize=(10, 5))
    dist = boot_result["distribution"]
    ax.hist(dist, bins=80, density=True, alpha=0.6, color="steelblue", edgecolor="white")
    ax.axvline(boot_result["observed"], color="red", linestyle="--", lw=2, label="Observed")
    ax.axvline(boot_result["ci_low"], color="orange", linestyle=":", lw=1.5, label=f"CI low")
    ax.axvline(boot_result["ci_high"], color="orange", linestyle=":", lw=1.5, label=f"CI high")
    ax.set_title(title)
    ax.set_xlabel("Estimate")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# Hypothesis testing result visualization
# ---------------------------------------------------------------------------

def plot_pay_gap_results(
    results_df: pd.DataFrame,
    save_as: str | None = None,
) -> plt.Figure:
    """
    Forest plot of gender pay gap by position with confidence indicators.
    """
    if results_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No results to display", ha="center")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Raw gap with significance markers
    ax = axes[0]
    order = results_df.sort_values("raw_gap")
    colors = ["#e74c3c" if r else "#2ecc71" for r in order["reject_adjusted"]]
    ax.barh(order["position"], order["raw_gap"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Raw Pay Gap (Male - Female)")
    ax.set_title("Gender Pay Gap by Position\n(red = significant after Holm correction)")

    # Panel 2: Effect sizes
    ax2 = axes[1]
    ax2.barh(order["position"], order["cohens_d"].abs(), color="steelblue", edgecolor="white")
    ax2.axvline(0.2, color="gray", linestyle="--", lw=0.8, label="|d| = 0.2 (small)")
    ax2.axvline(0.5, color="gray", linestyle="-.", lw=0.8, label="|d| = 0.5 (medium)")
    ax2.axvline(0.8, color="gray", linestyle=":", lw=0.8, label="|d| = 0.8 (large)")
    ax2.set_xlabel("|Cohen's d|")
    ax2.set_title("Effect Size by Position")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, save_as)
    return fig


def plot_anova_summary(
    tukey_df: pd.DataFrame,
    title: str = "Tukey HSD Pairwise Comparisons",
    save_as: str | None = None,
) -> plt.Figure:
    """Heatmap of Tukey HSD rejection matrix."""
    groups = sorted(set(tukey_df.iloc[:, 0].tolist() + tukey_df.iloc[:, 1].tolist()))
    matrix = pd.DataFrame(np.nan, index=groups, columns=groups)

    for _, row in tukey_df.iterrows():
        g1, g2 = row.iloc[0], row.iloc[1]
        # Use mean difference as value
        matrix.loc[g1, g2] = row.iloc[2]  # meandiff
        matrix.loc[g2, g1] = -row.iloc[2]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix.astype(float), annot=True, fmt=".0f",
        cmap="RdBu_r", center=0, square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# Compensation decomposition stacked bar
# ---------------------------------------------------------------------------

def plot_compensation_decomposition(
    decomp_df: pd.DataFrame,
    pct_cols: Sequence[str] = ("Salary_pct", "HA_pct", "COLA_pct", "CPF_pct"),
    title: str = "Compensation Structure by Position",
    save_as: str | None = None,
) -> plt.Figure:
    """Stacked bar chart showing percentage breakdown of compensation."""
    available = [c for c in pct_cols if c in decomp_df.columns]
    if not available:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No decomposition data available", ha="center")
        return fig

    fig, ax = plt.subplots(figsize=(12, 6))
    decomp_df[available].plot.bar(stacked=True, ax=ax, colormap="Set2", edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel("% of Gross Salary")
    ax.set_xlabel("Position")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    fig.tight_layout()
    _save(fig, save_as)
    return fig
