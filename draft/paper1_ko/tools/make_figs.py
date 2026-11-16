"""Figures of the manuscript from the evaluation outputs.

    python3 make_figs.py <dir with main.json and baselines.json> <output dir>
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

SRC, OUT = sys.argv[1], sys.argv[2]
for f in ("segoeui.ttf", "segoeuib.ttf"):
    fm.fontManager.addfont(f"/mnt/c/Windows/Fonts/{f}")
SURFACE, INK, INK2, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
plt.rcParams.update(
    {
        "font.family": "Segoe UI",
        "font.size": 11,
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "text.color": INK,
    }
)
# Categorical slots 1-4 of the reference palette, in fixed order; line style is the second encoding for print.
SERIES = [
    ("P1", "List-based method (P1)", "#2a78d6", "-"),
    ("P2-binary", "P2, yes-or-no answers", "#eb6834", (0, (6, 2))),
    ("P2-value", "P2, answer values", "#1baf7a", (0, (2, 1.5))),
    ("B-BED", "BED re-implementation", "#eda100", (0, (7, 2, 1.5, 2))),
]


def load(name):
    with open(f"{SRC}/{name}") as f:
        return json.load(f)


main, base = load("main.json"), load("baselines.json")
runs = {k: main[k] for k in ("P1", "P2-binary", "P2-value")}
runs["B-BED"] = base["B-BED"]["stats_file"]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for key, label, color, style in SERIES:
    curve = runs[key]["curve"]
    y = [100 * v for v in curve["gtpa1"]]
    ax.plot(curve["budgets"], y, color=color, lw=2, linestyle=style, solid_capstyle="round", label=label)
ax.set_xlim(0, 30)
ax.set_ylim(20, 100.5)
ax.set_xticks(range(0, 31, 5))
ax.set_xlabel("Number of questions after the initial evidence")
ax.set_ylabel("Accuracy for the true disease (%)")
ax.legend(loc="lower right", frameon=False, handlelength=3.2)
fig.savefig(f"{OUT}/figure1_accuracy_by_questions.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/figure1_accuracy_by_questions.pdf", bbox_inches="tight")
plt.close(fig)
print("written")
