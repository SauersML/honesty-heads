"""The replication figure: other ways of steering toward honesty against the 29 heads.
Left: held-out role-play lies on which the truthful answer becomes more likely than the lie, per steer, against the
range of that steer's random directions (same units, same per-unit sizes). Middle: per steer, the correlation over the
135 persona traits between its change and the 29 heads' change, after each direction's tendency to push every answer
further the way the model already leans is removed (persona_compare.py's beyond-confidence view), against the same for
its random directions. Right: the 29 heads' change per trait against the unsteered model's lean, with the 39 random
directions' least-squares lines.
  python code/replication_figure.py results/replication/compare.json results/persona.json replication.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import persona_compare as PC  # noqa: E402

cmp = json.load(open(sys.argv[1]))
ref = json.load(open(sys.argv[2]))
A = cmp["arms"]
ROWS = [("every head", "every head"), ("every head but the 29", "every head but the 29"),
        ("attribution patching head set", "attribution-patching heads\n(19 of its 20 in the 29)"),
        ("every neuron", "every MLP neuron"), ("neuron set", "175 MLP neurons"),
        ("residual stream layer 8", "residual stream, layer 8"), ("residual stream layer 16", "residual stream, layer 16"),
        ("residual stream layer 24", "residual stream, layer 24")]
ROWS = [(k, lab) for k, lab in ROWS if k in A]
STEER, GREY, BAND, INK = "#2E8B57", "#8A8A8A", "#D9D9D9", fs.INK
TT, LB, TK, TX = 50, 44, 42, 40
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": TK, "axes.labelsize": LB, "xtick.labelsize": TK,
                     "ytick.labelsize": TK, "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK})
n = len(ROWS) + 1
fig = plt.figure(figsize=(46, 3.0 + 1.75 * n))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.85, 1.1], wspace=0.42)
ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])


def bars(ax, vals, title, xlabel, lim):
    for i, (v, lo, hi, col) in enumerate(vals[::-1]):
        if v is None:
            ax.text(0.03 * (lim[1] - lim[0]), i, "reference", va="center", fontsize=TX, color=GREY)
            continue
        if lo is not None:
            ax.barh(i, hi - lo, left=lo, height=0.86, color=BAND, zorder=0)
        ax.barh(i, v, height=0.5, color=col, zorder=1)
        ax.text(max(v, 0) + 0.02 * (lim[1] - lim[0]), i, f"{v:.2f}", va="center", fontsize=TX, zorder=2)
    ax.axvline(0, color=INK, lw=1.5)
    ax.set_ylim(-0.6, len(vals) - 0.4)
    ax.set_xlim(*lim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=TT, fontweight="bold", loc="left", pad=28)
    fs.clean(ax)


heads_ref = json.load(open(Path(sys.argv[2]).parent / "heads_nat.json"))["test"]["rp lie"]
lv = [(heads_ref["set"]["removed"][0], None, None, "#2F6DB5")]
for k, _ in ROWS:
    x = A[k]["lies"]["rp lie"]
    r = x["random"]
    lv.append((x["share"], min(r) if r else None, max(r) if r else None, STEER))
labels = ["the 29 heads"] + [lab for _, lab in ROWS]
bars(ax1, lv, "Removes held-out role-play lies", "fraction of lies where the truth\nbecomes more likely than the lie",
     (0, 1))
ax1.set_yticks(range(n))
ax1.set_yticklabels(labels[::-1])

rv = [(None, None, None, None)]
for k, _ in ROWS:
    b = A[k]["beyond"]
    rv.append((b["r"], min(b["r_random"]) if b["r_random"] else None, max(b["r_random"]) if b["r_random"] else None,
               STEER))
bars(ax2, rv, "Same self-description changes\n(confidence effect removed)",
     "correlation with the 29 heads' changes\nover 135 traits", (-1, 1.25))
ax2.set_yticks(range(n))
ax2.set_yticklabels([])
ax2.set_xticks([-1, -0.5, 0, 0.5, 1])

traits, T, e = PC.trait_T(ref)
nm = ref["names"]
s = nm.index("set")
X = np.c_[np.ones(len(e)), e]
xs = np.linspace(e.min(), e.max(), 50)
for j in [j for j, v in enumerate(nm) if v.startswith("random")]:
    c = np.linalg.lstsq(X, T[:, j], rcond=None)[0]
    ax3.plot(xs, c[0] + c[1] * xs, color=GREY, lw=2, alpha=0.45, zorder=0)
c = np.linalg.lstsq(X, T[:, s], rcond=None)[0]
ax3.scatter(e, T[:, s], s=420, color="#2F6DB5", alpha=0.75, edgecolor="white", lw=1.5, zorder=2)
ax3.plot(xs, c[0] + c[1] * xs, color="#2F6DB5", lw=6, zorder=3)
ax3.axhline(0, color=INK, lw=1.5)
ax3.axvline(0, color=INK, lw=1.5)
ax3.set_xlabel("unsteered lean toward the trait\n(log2 odds of the trait-matching answer)")
ax3.set_ylabel("change under the 29 heads (bits)")
ax3.set_title("29 heads: answers move further\nthe way the model already leans", fontsize=TT, fontweight="bold",
              loc="left", pad=28)
fs.clean(ax3)
fig.legend(handles=[Patch(color=BAND, label="range of that steer's random directions"),
                    plt.Line2D([], [], color=GREY, lw=4, label="random directions (right)")],
           loc="upper left", bbox_to_anchor=(0.14, -0.01), ncol=2, frameon=False, fontsize=TX)
fig.savefig(sys.argv[3], dpi=110, bbox_inches="tight", pad_inches=0.4)
