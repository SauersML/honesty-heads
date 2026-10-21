"""Data for the one-page summary figure (summary_figure/figure.html -> summary.png). Reads results/ and data/, writes
summary_figure/data.js (window.DATA).
  python code/summary_figure/prepare.py"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import answer_split as AS  # noqa: E402
import corefind as C  # noqa: E402

R = ROOT / "results"
heads = json.load(open(R / "heads_nat.json"))
syco = json.load(open(R / "replication/sycophancy/heads_nat.json"))
bal = json.load(open(R / "replication/balanced/heads_nat.json"))
cmp = json.load(open(R / "replication/compare.json"))
ps = json.load(open(R / "persona_stats.json"))
lies = json.load(open(R / "replication/heads/lies.json"))
items = json.load(open(ROOT / "data/items_roleplay_insider.json"))
items = items["items"] if isinstance(items, dict) else items
out = {}

# where: the 29 heads and the heads found from sycophantic lies, on the 896-head grid
out["heads29"] = [[h["layer"], h["index"]] for h in heads["heads"]]
out["headsSyco"] = [[h["layer"], h["index"]] for h in syco["heads"]]
out["layers"] = [{"layer": l, "heads": 16 if l % 4 == 3 else 32} for l in range(32)]


def rnd_max(test, key="removed"):
    return max(v[key][0] for k, v in test.items() if k.startswith("random"))


# unseen role-play lies on which the truthful answer becomes more likely than the lie
t, tb = heads["test"]["rp lie"], bal["test"]["rp lie"]
out["flip"] = [
    {"label": "29 heads", "v": t["set"]["removed"], "n": t["set"]["n"], "rnd": rnd_max(t)},
    {"label": "50 heads, direction rebuilt with Yes and No lies balanced", "v": tb["set"]["removed"],
     "n": tb["set"]["n"], "rnd": rnd_max(tb)}]

# per lie, change toward the truthful answer under all-head steering, by what the lie says
f = lies["families"]["rp lie"]
m0, S = np.array(f["m"]), np.array(f["steered"])
cand = C.seeded([it for it in items if it["family"] == "rp" and it["split"] == "test" and it.get("label") == "lie"
                 and it.get("usable")], "rp/test/lie")
assert len(cand) == len(m0)
pol = [AS.polarity(it["lie"]) for it in cand]
names = lies["names"][1:]
j = names.index("every head")
rj = [k for k, n in enumerate(names) if n.startswith("every head/random")]
drop = m0[:, None] - S
out["polar"] = [{"pol": p, "steer": round(float(drop[i, j]), 3), "random": drop[i, rj].round(3).tolist()}
                for i, p in enumerate(pol)]

# self-descriptions: traits significant (permutation FDR 5%), moving the same way on Yes- and No-matching questions,
# and still significant once the confidence trend is removed
qb = cmp["reference"]["q_beyond"]
tr = [x for x in ps["traits"] if x["q_perm"] <= 0.05 + 1e-9 and x["both_halves"] and qb[x["trait"]] <= 0.05 + 1e-9]
out["traits"] = [{"trait": x["trait"], "T": x["T_set"], "lo": min(x["T_random"]), "hi": max(x["T_random"])}
                 for x in sorted(tr, key=lambda x: -x["T_set"])]
out["n_traits"] = len(ps["traits"])

# other ways of steering: lie flips and agreement with the 29 heads' self-description changes
A = cmp["arms"]
rows = [("every head", "all 896 heads"), ("every head but the 29", "all heads except the 29"),
        ("neuron set", "175 MLP neurons"), ("every neuron", "all MLP neurons"),
        ("residual stream layer 24", "residual stream, layer 24")]
out["methods"] = []
for k, lab in rows:
    x, b = A[k]["lies"]["rp lie"], A[k]["beyond"]
    out["methods"].append({"label": lab, "flip": x["share"], "flipHi": max(x["random"]) if x["random"] else None,
                           "flipLo": min(x["random"]) if x["random"] else None, "r": b["r"],
                           "rLo": min(b["r_random"]) if b["r_random"] else None,
                           "rHi": max(b["r_random"]) if b["r_random"] else None})
open(Path(__file__).parent / "data.js", "w").write("window.DATA = " + json.dumps(out) + ";\n")
print({k: (len(v) if isinstance(v, list) else v) for k, v in out.items()})
