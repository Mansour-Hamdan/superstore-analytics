"""Shared chart styling."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, RED, GRAY = "#2a78d6", "#eb6834", "#e34948", "#b5b3ad"
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "font.size": 10,
        "figure.dpi": 150,
    }
)


def comparison_bars(ax, names, values, highlight, fmt):
    """Horizontal bars; the original RapidMiner replica is drawn in gray."""
    colors = [GRAY if n == highlight else BLUE for n in names]
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.55)
    ax.bar_label(bars, fmt=fmt, padding=4, color=INK)
    ax.tick_params(axis="y", colors=INK, length=0)
    return bars
