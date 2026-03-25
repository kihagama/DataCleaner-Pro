"""
Visualization Module
====================
Generates static (Matplotlib/Seaborn) and interactive (Plotly) charts.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns

logger = logging.getLogger(__name__)

# ── colour palette ──────────────────────────────────────────
PALETTE = px.colors.qualitative.Vivid


# ──────────────────────────────────────────────────────────────
# Static helpers (used when generating an HTML/PDF report)
# ──────────────────────────────────────────────────────────────

def _savefig(fig, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def plot_histogram_static(df: pd.DataFrame, col: str, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    df[col].dropna().hist(bins=30, ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_title(f"Distribution — {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    return _savefig(fig, f"{out_dir}/hist_{col}.png")


def plot_boxplot_static(df: pd.DataFrame, col: str, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 3))
    df[col].dropna().plot.box(ax=ax, vert=False, patch_artist=True,
                              boxprops=dict(facecolor="#4C72B0", color="#2d4070"))
    ax.set_title(f"Boxplot — {col}")
    return _savefig(fig, f"{out_dir}/box_{col}.png")


def plot_bar_static(df: pd.DataFrame, col: str, out_dir: str, top_n: int = 15) -> str:
    counts = df[col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.bar(ax=ax, color=sns.color_palette("viridis", len(counts)))
    ax.set_title(f"Top {top_n} Values — {col}")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    return _savefig(fig, f"{out_dir}/bar_{col}.png")


def plot_correlation_heatmap_static(df: pd.DataFrame, out_dir: str) -> Optional[str]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(max(6, len(corr)), max(5, len(corr) - 1)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    return _savefig(fig, f"{out_dir}/correlation_heatmap.png")


# ──────────────────────────────────────────────────────────────
# Interactive Plotly charts (used in Streamlit dashboard)
# ──────────────────────────────────────────────────────────────

def plotly_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(df, x=col, nbins=40,
                       title=f"Distribution — {col}",
                       color_discrete_sequence=["#4361EE"])
    fig.update_layout(bargap=0.05)
    return fig


def plotly_boxplot(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.box(df, y=col, title=f"Boxplot — {col}",
                 color_discrete_sequence=["#3A0CA3"])
    return fig


def plotly_bar(df: pd.DataFrame, col: str, top_n: int = 15) -> go.Figure:
    counts = df[col].value_counts().head(top_n).reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(counts, x=col, y="count",
                 title=f"Top {top_n} Values — {col}",
                 color="count",
                 color_continuous_scale="Viridis")
    return fig


def plotly_pie(df: pd.DataFrame, col: str, top_n: int = 10) -> go.Figure:
    counts = df[col].value_counts().head(top_n)
    fig = px.pie(values=counts.values, names=counts.index,
                 title=f"Pie Chart — {col}",
                 color_discrete_sequence=PALETTE)
    return fig


def plotly_correlation_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(title="Correlation Heatmap",
                      xaxis_tickangle=-45)
    return fig


def plotly_missing_bar(report: dict) -> go.Figure:
    cols  = [c["column"]   for c in report["columns"] if c["null_count"] > 0]
    nulls = [c["null_pct"] for c in report["columns"] if c["null_count"] > 0]
    if not cols:
        fig = go.Figure()
        fig.update_layout(title="No Missing Values 🎉")
        return fig
    fig = px.bar(x=cols, y=nulls,
                 labels={"x": "Column", "y": "% Missing"},
                 title="Missing Values by Column (%)",
                 color=nulls,
                 color_continuous_scale="Reds")
    return fig


def plotly_outlier_bar(report: dict) -> go.Figure:
    cols    = [c["column"]      for c in report["columns"] if c["outliers_iqr"] > 0]
    counts  = [c["outliers_iqr"] for c in report["columns"] if c["outliers_iqr"] > 0]
    if not cols:
        fig = go.Figure()
        fig.update_layout(title="No IQR Outliers Detected 🎉")
        return fig
    fig = px.bar(x=cols, y=counts,
                 labels={"x": "Column", "y": "Outlier Count"},
                 title="Outliers per Column (IQR method)",
                 color=counts,
                 color_continuous_scale="Oranges")
    return fig
