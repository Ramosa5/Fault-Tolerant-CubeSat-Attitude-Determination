"""Publication-style visualization helpers inspired by figures4papers.

Conventions adapted from:
https://github.com/ChenLiu-1996/figures4papers

The upstream repository documents style/API conventions rather than shipping a
single importable plotting package. This module implements the relevant ideas
locally so experiment runs are self-contained and reproducible.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#4D4D4D",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}
DEFAULT_COLORS = [
    PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"],
    PALETTE["teal"], PALETTE["violet"], PALETTE["neutral_dark"],
]

@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 15
    axes_linewidth: float = 2.0
    use_tex: bool = False
    font_family: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    style = style or FigureStyle()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(style.font_family),
        "font.size": style.font_size,
        "axes.linewidth": style.axes_linewidth,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": style.use_tex,
        "lines.linewidth": 2.2,
        "axes.labelpad": 7,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
    })


def create_subplots(nrows=1, ncols=1, figsize=None, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, np.atleast_1d(axes).ravel()


def _clean_basename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return re.sub(r"_+", "_", name).strip("_") or "figure"


def finalize_figure(fig, out_path, formats=("png", "pdf"), dpi=300, close=True, pad=0.6):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_name(_clean_basename(out_path.name))
    saved = []
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".")
        target = stem.with_suffix("." + fmt)
        fig.savefig(target, dpi=dpi, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        saved.append(target)
    if close:
        plt.close(fig)
    return saved


def make_trend(ax, x, y_series, labels, colors=None, ylabel=None, xlabel=None, show_shadow=False):
    x = np.asarray(x)
    colors = colors or DEFAULT_COLORS
    for i, (y, label) in enumerate(zip(y_series, labels)):
        y = np.asarray(y)
        ax.plot(x, y, label=label, color=colors[i % len(colors)])
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax


def make_grouped_bar(ax, categories, series, labels, ylabel="Value", colors=None, annotate=False):
    colors = colors or DEFAULT_COLORS
    categories = list(categories); series = [np.asarray(s, float) for s in series]
    x = np.arange(len(categories)); width = 0.78 / max(1, len(series)); last = None
    for i, (vals, label) in enumerate(zip(series, labels)):
        pos = x + (i - (len(series)-1)/2)*width
        last = ax.bar(pos, vals, width=width, label=label, color=colors[i % len(colors)], edgecolor="black", linewidth=1.0)
        if annotate: annotate_bars(ax, last)
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    return last


def annotate_bars(ax, bars, fmt="{:.2f}", fontsize=9, padding=3):
    for bar in bars:
        v = bar.get_height()
        if np.isfinite(v):
            ax.annotate(fmt.format(v), (bar.get_x()+bar.get_width()/2, v), xytext=(0,padding), textcoords="offset points", ha="center", va="bottom", fontsize=fontsize)


def make_heatmap(ax, matrix, x_labels=None, y_labels=None, cmap="Blues", cbar_label=None, annotate=False, fmt="d"):
    matrix = np.asarray(matrix)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)), x_labels, rotation=35, ha="right")
    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)), y_labels)
    if annotate:
        threshold = np.nanmax(matrix) * 0.55 if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i,j]
                txt = format(int(value), fmt) if fmt == "d" else format(value, fmt)
                ax.text(j, i, txt, ha="center", va="center", color="white" if value > threshold else "black", fontsize=9)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label: cbar.set_label(cbar_label)
    return im


def add_fault_span(ax, start, end, label="Fault active"):
    ax.axvspan(start, end, color=PALETTE["red_1"], alpha=0.35, lw=0, label=label)


def save_line_figure(out_path, x, series, labels, xlabel, ylabel, title=None, formats=("png","pdf"), dpi=300, xscale=None, yscale=None, legend=True, fault_span=None):
    fig, axes = create_subplots(figsize=(8.2, 4.8)); ax=axes[0]
    make_trend(ax, x, series, labels, xlabel=xlabel, ylabel=ylabel)
    if title: ax.set_title(title)
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)
    if fault_span is not None: add_fault_span(ax, fault_span[0], fault_span[1])
    if legend and len(labels): ax.legend()
    ax.margins(x=0.01)
    fig.tight_layout(pad=1.2)
    return finalize_figure(fig, out_path, formats=formats, dpi=dpi)
