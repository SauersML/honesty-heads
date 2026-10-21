"""One landscape figure for the role-play honesty heads of Qwen3.5-9B (inputs in results/ and data/).
Left, top: 34 lies from held-out role-play scenarios (answer token, exact 95% intervals; "all but the 29" from the
  float32 replication run, results/replication/heads/lies.json). Left, middle: insider trading, 45 conversations in which the model lies to its
  manager, full 512-token replies graded by audit_eval's rules (lie, deny_tip = a denial that names the tip or its
  source, truth, other = both, neither or no complete message).
Left, bottom: path patching on the 34 lies (dag_nat.json). Box = a head steered alone (its share of the steering's
  effect, median over lies); arrow from a prompt part = the head steered only at that part's tokens; head -> head = the
  share of the first head's effect that passes through the second; head -> logit = with every later chosen head clamped
  unsteered. Drawn: the 10 largest heads, arrows whose exact 95% interval excludes 0 and whose median is at least 2%
  (prompt parts, logit) or 0.5% (head -> head); width ~ share; edges translucent. Bold lines: J-lens themes (natjlens).
Right: Anthropic's persona evals (persona_stats.py; 16 questions per trait, 8 where "Yes" matches the behaviour and 8
  where "No" does), trait names as the dataset's file names: change = log2 odds of answer_matching_behavior, both
  halves weighted equally. Shown: every trait with permutation-FDR q <= 0.05 over 39 random directions (same heads,
  same per-head sizes). Black outline: also robust to acquiescence bias (after subtracting each direction's median
  tilt toward "Yes", both halves move the same way, each beyond all but one random direction).
  python code/figure.py results/persona_stats.json honesty_heads.png"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs  # noqa: E402

fs.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import beta  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
HERE = os.path.join(ROOT, "data")
S = json.load(open(sys.argv[1]))
natsel = json.load(open(f"{R}/heads_nat.json"))
rep_heads = json.load(open(f"{R}/replication/heads/lies.json"))
dag = json.load(open(f"{R}/dag_nat.json"))
itr = json.load(open(f"{R}/free_core_rp29_it.json"))
ROLE = json.load(open(os.path.join(HERE, "honesty_roles.json")))
T2, LB, TK, TX, TT = 66, 58, 54, 54, 46          # panel title, axis label, tick, text, trait labels
STEER, TRUE, LIE, GREY, BAND, INK = fs.BLUE, fs.GREEN, fs.RED, "#9AA5B1", "#DCDCDC", fs.INK
plt.rcParams.update({"font.size": TX, "axes.labelsize": LB, "xtick.labelsize": TK, "ytick.labelsize": TK})
tr = S["traits"]
sig_ = sorted([r for r in tr if r["q_perm"] <= 0.05 + 1e-9], key=lambda r: r["T_set"])   # largest at the top
PITCH = 1.0                                         # inches per trait row
HP = PITCH * len(sig_)
FW, FH = 72.0, 4.2 + HP + 3.6
WL = 35.0                                           # left column width
fig = plt.figure(figsize=(FW, FH))


def axin(x, top, w, h):            # axes by inches from the left and from the top
    return fig.add_axes([x / FW, 1 - (top + h) / FH, w / FW, h / FH])


def txt(x, y, s, **kw):            # text by inches from the left and from the top
    kw.setdefault("va", "top")
    return fig.text(x / FW, 1 - y / FH, s, **kw)


def title(x, y, s):
    txt(x, y, s, fontsize=T2, fontweight="bold", color=INK, linespacing=1.2)


def name(t):                       # the dataset's own trait name, hyphens as spaces
    return t.replace("non-HHH", "non\0HHH").replace("-", " ").replace("\0", "-")


# ---------------------------------------------------------------- left: the circuit (graphviz)
EG, HL = dag["families"]["rp lie"]["edges"], dag["heads"]


def val(k):
    return 100 * EG[k][0]


def sig(k):
    return EG[k][1] > 0 or EG[k][2] < 0


idx = sorted([i for i in range(len(HL)) if sig(f"head {i}") and EG[f"head {i}"][0] > 0],
             key=lambda i: -EG[f"head {i}"][0])[:10]
nid = {i: HL[i].replace(".", "_") for i in idx}
lay = sorted({int(HL[i][1:].split(".")[0]) for i in idx})
F = "Helvetica Neue"
dot = ['digraph G {', 'rankdir=TB; newrank=true; splines=true; nodesep=0.5; ranksep=0.9; pad=0.05;',
       f'bgcolor="white"; fontname="{F}";',
       f'node [shape=box, style="rounded,filled", fontname="{F}", penwidth=0, margin="0.2,0.12"];',
       'edge [color="#5A5A5ABB", arrowsize=1.8];']
SRC = [("system", "system prompt"), ("user", "user's question"), ("reply", "start of reply")]
for g, lab in SRC:
    dot.append(f'src_{g} [label="{lab}", fillcolor="{STEER}", fontcolor="white", fontsize=64];')
dot.append('{rank=min; ' + " ".join(f"src_{g};" for g, _ in SRC) + '}')
for l in lay:
    ns = [i for i in idx if int(HL[i][1:].split(".")[0]) == l]
    for i in ns:
        role = ROLE.get(HL[i], "").replace("\n", "<BR/>")
        sub = f'{HL[i]}  {val(f"head {i}"):.0f}%'
        lab = f'<<B>{role}</B><BR/>{sub}>' if role else f'<{sub}>'
        dot.append(f'{nid[i]} [label={lab}, fillcolor="{TRUE}", fontcolor="white", fontsize=62];')
    dot.append('{rank=same; ' + " ".join(f"{nid[i]};" for i in ns) + '}')
dot.append(f'answer [label="truthful answer", fillcolor="{INK}", fontcolor="white", fontsize=64];')
dot.append('{rank=max; answer;}')


def pw(v):
    return f"{2.6 + 2.0 * min(abs(v), 6):.2f}"


for i in idx:
    for g, _ in SRC:
        k = f"src {g} {i}"
        if sig(k) and val(k) >= 2.0:
            dot.append(f'src_{g} -> {nid[i]} [penwidth={pw(val(k))}, color="{STEER}BB"];')
    k = f"direct {i}"
    if sig(k) and val(k) >= 2.0:
        dot.append(f'{nid[i]} -> answer [penwidth={pw(val(k))}];')
for k in EG:
    if k.startswith("via "):
        i, j = map(int, k.split()[1:])
        if i in idx and j in idx and sig(k) and val(k) >= 0.5:
            dot.append(f'{nid[i]} -> {nid[j]} [penwidth={pw(3 * val(k))}, color="{TRUE}BB"];')
dot.append('}')
tmp = tempfile.mkdtemp()
open(f"{tmp}/c.dot", "w").write("\n".join(dot))
subprocess.run(["dot", "-Tpng", "-Gdpi=170", f"{tmp}/c.dot", "-o", f"{tmp}/c.png"], check=True)
img = plt.imread(f"{tmp}/c.png")
wi, hi_ = img.shape[1] / 170, img.shape[0] / 170
sc = min(1.25, WL / wi, (FH - 13.4 - 2 * 8.0) / hi_)
print(f"circuit natural {wi:.1f} x {hi_:.1f} in, drawn at scale {sc:.2f}")
H1 = (FH - hi_ * sc - 13.4) / 2

# ---------------------------------------------------------------- left: test sets
y = 0.3
t = natsel["test"]["rp lie"]
title(0.5, y, "Test set: role-play lies (scenarios unseen in head search)")
y += 1.6
ax = axin(7.4, y, WL - 10.0, H1)
rnd = [t[x]["removed"][0] for x in t if x.startswith("random")]
f = rep_heads["families"]["rp lie"]
k, n = int((np.array(f["steered"])[:, rep_heads["names"].index("every head but the 29") - 1] < 0).sum()), len(f["m"])
nd = (k / n, beta.ppf(.025, k, n - k + 1) if k else 0.0, beta.ppf(.975, k + 1, n - k) if k < n else 1.0)
rows = [("the 29 heads", *t["set"]["removed"], STEER), ("all 896 heads", *t["every_head"]["removed"], GREY),
        ("all but the 29", *nd, GREY), ("29 random heads", max(rnd), min(rnd), max(rnd), "#C8C8C8")]
for i, (nm, v, lo, hi, col) in enumerate(rows[::-1]):
    ax.barh(i, max(v, 0.01), height=0.66, color=col)
    if nm.startswith("29 random"):
        ax.text(hi + 0.02, i, f"≤ {hi:.2f}", va="center", fontsize=TX)
        continue
    ax.plot([lo, hi], [i, i], color=INK, lw=5)
    ax.text(hi + 0.02, i, f"{v:.2f}", va="center", fontsize=TX)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.set_ylim(-0.5, len(rows) - 0.5)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xlabel("fraction of lies where the truth\nbecomes more likely than the lie")
fs.clean(ax)
y += H1 + 3.3

core = [s for s in itr["sets"] if s["set"] == "core"][0]
rs = [s for s in itr["sets"] if s["set"] != "core"]
n_it = len(core["grades"])
title(0.5, y, "Test set: insider trading (model lies to its manager)")
y += 1.6
ax = axin(7.4, y, WL - 7.4, H1)
conds = [("unsteered", ["lie"] * n_it), ("steered", core["grades"])] + \
        [(f"random dir. {k + 1}", s["grades"]) for k, s in enumerate(rs)]
cats = [("lie", "lie", LIE), ("deny_tip", "denies, names the tip", "#E8A33D"), ("other", "other", "#CFCFCF"),
        ("truth", "truth", TRUE)]
for i, (nm, g) in enumerate(conds[::-1]):
    left = 0
    for k, lab, col in cats:
        w = sum(x == k for x in g)
        ax.barh(i, w, left=left, height=0.7, color=col)
        if w >= 3:
            ax.text(left + w / 2, i, str(w), ha="center", va="center", fontsize=TX,
                    color="white" if col in (LIE, TRUE) else INK)
        left += w
ax.set_yticks(range(len(conds)))
ax.set_yticklabels([c[0] for c in conds[::-1]])
ax.set_ylim(-0.5, len(conds) - 0.5)
ax.set_xlim(0, n_it)
ax.set_xlabel("conversations")
fs.clean(ax)
y += H1 + 1.9
fig.legend(handles=[Patch(color=c, label=l) for _, l, c in cats], loc="upper left", bbox_to_anchor=(0.5 / FW, 1 - y / FH),
           ncol=4, fontsize=TX - 4, frameon=False, handlelength=1.0, columnspacing=1.2, handletextpad=0.4)
y += 2.0

# ---------------------------------------------------------------- left: the circuit (drawn)
title(0.5, y, "Circuit of the steering effect (path patching)")
y += 1.5
ax = axin(0.5, y, wi * sc, hi_ * sc)
ax.imshow(img, interpolation="lanczos")
ax.axis("off")
y += hi_ * sc + 0.3
fig.legend(handles=[Line2D([], [], color=STEER, lw=10, label=r"prompt $\rightarrow$ head"),
                    Line2D([], [], color=TRUE, lw=10, label=r"head $\rightarrow$ head (indirect)"),
                    Line2D([], [], color="#5A5A5A", lw=10, label=r"head $\rightarrow$ logit (direct)")],
           loc="upper left", bbox_to_anchor=(0.5 / FW, 1 - y / FH), ncol=3, fontsize=TX - 6, frameon=False,
           handlelength=1.4, columnspacing=1.4, handletextpad=0.5)

# ---------------------------------------------------------------- right: every significant persona trait
X0 = 36.6
title(X0, 0.3, f"Anthropic persona evals: all {len(sig_)} of {len(tr)} traits that change significantly\n"
      f"vs random directions (permutation FDR 5%)")
AX = X0 + 21.0
ax = axin(AX, 4.2, FW - 0.5 - AX, HP)
for i, r in enumerate(sig_):
    rr = np.array(r["T_random"])
    ax.barh(i, rr.max() - rr.min(), left=rr.min(), height=0.86, color=BAND, zorder=1)
    ax.barh(i, r["T_set"], height=0.6, color=LIE if r["T_set"] > 0 else STEER, zorder=2,
            edgecolor=INK if r["both_halves"] else "none", linewidth=9)
ax.axvline(0, color=INK, lw=2.5, zorder=3)
ax.set_yticks(range(len(sig_)))
ax.set_yticklabels([name(r["trait"]) for r in sig_], fontsize=TT)
ax.set_ylim(-0.6, len(sig_) - 0.4)
lim = max(abs(np.array([x for r in sig_ for x in r["T_random"]] + [r["T_set"] for r in sig_]))) * 1.03
ax.set_xlim(-lim, lim)
ax.set_xlabel("change in log2 odds of the\ntrait-matching answer (bits)")
ax.text(0.0, 1.004, "less of trait", transform=ax.transAxes, ha="left", va="bottom", color=STEER, fontsize=TX - 4)
ax.text(1.0, 1.004, "more of trait", transform=ax.transAxes, ha="right", va="bottom", color=LIE, fontsize=TX - 4)
fs.clean(ax)
fig.legend(handles=[Patch(facecolor="white", edgecolor=INK, lw=9, label="robust to acquiescence (yes) bias"),
                    Patch(color=BAND, label="range of 39 random directions")],
           loc="upper left", bbox_to_anchor=(X0 / FW, 1 - (4.2 + HP + 0.9) / FH), ncol=1, fontsize=TX - 4,
           frameon=False, handlelength=1.4, labelspacing=0.3)
fs.save(fig, sys.argv[2])
print(sys.argv[2])
