"""Is the honesty steer a push toward "Yes"? Splits each steer's effect on held-out role-play lies (and resists) by the
polarity of the answer the model gives (lies: the lie token; resists: the truth token): a yes-word, a no-word, or
anything else. With drop = m unsteered minus m steered (m = log2 p(lie) - log2 p(truth); resists under -alpha, sign
flipped so positive = toward the lie), a pure push toward "Yes" lowers m where the lie is a no-word and raises it where
the lie is a yes-word; an answer-blind honesty push lowers both. Per steer, against its random directions:
  honesty = (mean drop on no-lies + mean drop on yes-lies) / 2,  yes push = (mean drop on no-lies - on yes-lies) / 2,
  and the mean drop on lies whose answers are neither. Item order is persona_rep.py run's (nat_recs: seeded, usable);
  a family whose item count differs from the file's is skipped.
  python code/answer_split.py ITEMS.json OUT.json LIES.json [LIES.json ...]"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corefind as C  # noqa: E402

YES = {"yes", "yeah", "yep", "absolutely", "sure", "definitely", "certainly", "indeed"}
NO = {"no", "not", "nope", "never"}


def polarity(tokens: list) -> str:
    w = (tokens[0] if tokens else "").strip().lower()
    return "yes" if w in YES else "no" if w in NO else "other"


def split(m0: np.ndarray, S: np.ndarray, pol: np.ndarray, sign: float) -> dict:
    """Per column of S: the mean drop toward the truth (lies) or toward the lie (resists), by polarity."""
    drop = sign * (m0[:, None] - S) if sign > 0 else S - m0[:, None]
    o = {k: drop[pol == k].mean(0) if (pol == k).any() else np.full(S.shape[1], np.nan) for k in ("yes", "no", "other")}
    o["honesty"] = (o["no"] + o["yes"]) / 2 if sign > 0 else (o["yes"] + o["no"]) / 2
    o["yes_push"] = (o["no"] - o["yes"]) / 2 if sign > 0 else (o["yes"] - o["no"]) / 2
    o["share"] = {k: ((S < 0) if sign > 0 else (S > 0))[pol == k].mean(0) for k in ("yes", "no", "other")}
    return o


def main():
    items = json.load(open(sys.argv[1]))
    items = items["items"] if isinstance(items, dict) else items
    out = {}
    for path in sys.argv[3:]:
        L = json.load(open(path))
        for fam, lab, sign in (("rp", "lie", 1.0), ("rp", "resist", -1.0)):
            key = f"{fam} {lab}"
            if key not in L["families"]:
                continue
            cand = C.seeded([it for it in items if it["family"] == fam and it["split"] == "test" and
                             it.get("label") == lab and it.get("usable")], f"{fam}/test/{lab}")
            f = L["families"][key]
            m0, S = np.array(f["m"]), np.array(f["steered"])
            if len(cand) != len(m0):
                print(f"{path} {key}: {len(cand)} items vs {len(m0)} scored, skipped")
                continue
            pol = np.array([polarity(it["lie"] if lab == "lie" else it["truth"]) for it in cand])
            o = split(m0, S, pol, sign)
            names = L["names"][1:]
            for arm in L["arms"]:
                j = names.index(arm["name"])
                rj = [k for k, n in enumerate(names) if n.startswith(arm["name"] + "/random")]
                rec = {"n": {k: int((pol == k).sum()) for k in ("yes", "no", "other")}}
                for q in ("honesty", "yes_push", "yes", "no", "other"):
                    rec[q] = round(float(o[q][j]), 4)
                    if rj:
                        r = o[q][rj]
                        rec[q + "_random"] = r.round(4).tolist()
                        rec[q + "_p"] = round(float((1 + (r >= o[q][j]).sum()) / (len(r) + 1)), 4)
                rec["share"] = {k: round(float(v[j]), 4) for k, v in o["share"].items()}
                rec["share_random_max"] = {k: round(float(np.nanmax(v[rj])), 4) if rj else None
                                           for k, v in o["share"].items()}
                out.setdefault(arm["name"], {})[key] = rec
                print(f"{key:10s} {arm['name']:32s} honesty {rec['honesty']:+.2f} (p {rec.get('honesty_p', float('nan')):.3f}, "
                      f"random max {max(rec.get('honesty_random', [np.nan])):+.2f})  yes push {rec['yes_push']:+.2f} "
                      f"(random max {max(rec.get('yes_push_random', [np.nan])):+.2f})  other {rec['other']:+.2f}  "
                      f"removed yes {rec['share']['yes']:.2f} no {rec['share']['no']:.2f} other {rec['share']['other']:.2f}")
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
