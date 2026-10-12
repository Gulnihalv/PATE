"""Figure: accuracy against coset density, both solvers.

Drop into the evaluation notebook. The point of the figure is that cells with
very different (key length, text length) but the same characters-per-coset land
on the same place on the curve, so the marker shape encodes text length and the
colour encodes the solver: two markers of different shape sitting on one point
are two different configurations agreeing.
"""

import matplotlib.pyplot as plt
import numpy as np

PATE_COLOR = "#2a78d6"
CLS_COLOR = "#eb6834"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#ffffff"

# marker per text length, so matched cells stay countable where they overlap
MARKERS = {64: "o", 128: "s", 256: "^", 400: "v", 512: "D"}


def _points(results, field):
    """[(L/d, accuracy, text_len)] for every cell that has the field."""
    return sorted((L / k, cell[field], L)
                  for k, cols in results.items()
                  for L, cell in cols.items() if field in cell)


def _curve(pts):
    """One point per coset density, averaging the cells that share one.

    Cells at the same density are repeat measurements of the same quantity, so
    the curve averages them; leaving duplicates in would make the threshold
    read out at a measured density rather than between two of them.
    """
    agg = {}
    for x, y, _ in pts:
        agg.setdefault(round(x, 4), []).append(y)
    return sorted((x, sum(v) / len(v)) for x, v in agg.items())


def _crossing(pts, target):
    """Characters per coset at which the curve reaches `target`.

    Read by linear interpolation between the two measured densities that
    bracket it, after averaging cells that share a density.
    """
    curve = _curve(pts)
    xs = [x for x, _ in curve]
    ys, run = [], float("-inf")
    for _, y in curve:                        # monotone envelope: the curve is noisy
        run = max(run, y)
        ys.append(run)
    if ys[-1] < target:
        return None
    for i, y in enumerate(ys):
        if y >= target:
            if i == 0 or ys[i] == ys[i - 1]:
                return xs[i]
            f = (target - ys[i - 1]) / (ys[i] - ys[i - 1])
            return xs[i - 1] + f * (xs[i] - xs[i - 1])
    return None


def plot_density(neural, classical, field="oracle_char_acc", target=95,
                 out=None, figsize=(7.0, 4.4)):
    np_pts, cl_pts = _points(neural, field), _points(classical, field)

    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for pts, color, label in ((cl_pts, CLS_COLOR, "classical"),
                              (np_pts, PATE_COLOR, "PATE")):
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=color, lw=2, zorder=2, solid_capstyle="round")
        for L in sorted({p[2] for p in pts}):
            sel = [p for p in pts if p[2] == L]
            ax.plot([p[0] for p in sel], [p[1] for p in sel],
                    linestyle="none", marker=MARKERS.get(L, "o"), ms=7,
                    mfc=color, mec=SURFACE, mew=1.4, zorder=3)

    # the sample-efficiency claim, drawn rather than only stated
    if target is not None:
        xn, xc = _crossing(np_pts, target), _crossing(cl_pts, target)
        if xn and xc:
            y_arrow = 34
            ax.axhline(target, color=INK_SOFT, lw=0.9, ls=(0, (4, 3)), zorder=1)
            for x, color in ((xn, PATE_COLOR), (xc, CLS_COLOR)):
                ax.plot([x, x], [y_arrow, target], color=color, lw=0.9,
                        ls=(0, (2, 2)), zorder=1)
            ax.annotate("", xy=(xn, y_arrow), xytext=(xc, y_arrow),
                        arrowprops=dict(arrowstyle="<->", color=INK_SOFT, lw=0.9))
            mid = np.sqrt(xn * xc)
            ax.text(mid, y_arrow + 2.5, f"{xc / xn:.1f}\u00d7 fewer characters per coset",
                    ha="center", va="bottom", fontsize=8.5, color=INK_SOFT)
            ax.text(xn, y_arrow - 7, f"{xn:.1f}", ha="center", fontsize=8.5,
                    color=PATE_COLOR)
            ax.text(xc, y_arrow - 7, f"{xc:.1f}", ha="center", fontsize=8.5,
                    color=CLS_COLOR)
            ax.text(0.015, target + 1.5, f"{target}% accuracy", ha="left",
                    va="bottom", fontsize=8.5, color=INK_SOFT,
                    transform=ax.get_yaxis_transform())

    # direct labels, so identity is never carried by colour alone
    for pts, color, label, dy in ((np_pts, PATE_COLOR, "PATE", 4),
                                  (cl_pts, CLS_COLOR, "classical", -11)):
        x, y = pts[-4][0], pts[-4][1]
        ax.text(x, y + dy, label, color=color, fontsize=10, fontweight="bold",
                ha="center")

    ax.set_xscale("log")
    ax.set_xlabel("characters per coset   $L/d$", fontsize=10, color=INK)
    ax.set_ylabel("character accuracy (%)", fontsize=10, color=INK)
    ax.set_ylim(0, 103)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128"])
    ax.tick_params(labelsize=9, colors=INK_SOFT, length=3)
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)

    shapes = [plt.Line2D([], [], linestyle="none", marker=MARKERS.get(L, "o"),
                         ms=7, mfc=INK_SOFT, mec=SURFACE, mew=1.2,
                         label=f"{L} chars")
              for L in sorted({p[2] for p in np_pts})]
    leg = ax.legend(handles=shapes, loc="lower right", fontsize=8.5,
                    frameon=False, handletextpad=0.4, labelspacing=0.3,
                    title="ciphertext length", title_fontsize=8.5)
    leg.get_title().set_color(INK_SOFT)
    for t in leg.get_texts():
        t.set_color(INK_SOFT)

    fig.tight_layout()
    if out:
        fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    return fig, ax
