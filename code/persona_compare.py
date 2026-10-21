"""Does each honesty steer change the same self-descriptions as the 29 honesty heads? Inputs: the reference persona file
(steer.py natpersona: the 29-head set and its random directions) and persona_rep.py run outputs (persona.json and
lies.json per group). Per trait, T = the balanced change (persona_stats.py). Two views of it:
  raw: T itself;
  beyond confidence: T minus each direction's own least-squares line on the unsteered endorsement (the trait's mean m,
    base column), i.e. what is left once a direction's tendency to push every answer further the way the model already
    leans (or back toward even) is removed.
Per arm, with the arm's own random directions as the null (each scored the same way, p = rank among them), in each view:
  r over traits with the reference's T; the share of the reference's permutation-FDR traits (q <= .05 in that view)
  whose change has the same sign; the arm's own permutation-FDR count and its overlap with the reference's; for raw T,
  the slope on the unsteered endorsement and the dark and deception set means (persona_stats sets).
  lies: per natural family, the share of test lies whose m falls below 0 under +alpha (resists: above 0 under -alpha),
  with the arm's random directions' range; controls: the share the unedited model gets right that stay right.
  persona_compare.py REF_PERSONA.json OUT.json DIR [DIR ...]   (each DIR holds persona.json and lies.json)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import persona_stats as PS  # noqa: E402


def trait_T(rec: dict) -> tuple:
    """(traits, T [traits, columns], endorsement [traits]): per trait the balanced change against the base column, and
    the base column's mean m (positive: the unsteered model leans toward the trait)."""
    traits = sorted({it["trait"] for it in rec["items"]})
    m = {t: {"Yes": [], "No": []} for t in traits}
    for it in rec["items"]:
        m[it["trait"]][it["match"]].append(it["m"])
    T, e = [], []
    for t in traits:
        y, n = np.array(m[t]["Yes"]), np.array(m[t]["No"])
        T.append(((y - y[:, :1]).mean(0) + (n - n[:, :1]).mean(0)) / 2)
        e.append(np.concatenate([y[:, 0], n[:, 0]]).mean())
    return traits, np.array(T), np.array(e)


def beyond(A: np.ndarray, e: np.ndarray) -> np.ndarray:
    """A [traits, directions] minus each column's least-squares line on e."""
    X = np.c_[np.ones(len(e)), e]
    return A - X @ np.linalg.lstsq(X, A, rcond=None)[0]


def rank_p(x: float, null: np.ndarray, side: str = "greater") -> float:
    return float((1 + ((null >= x) if side == "greater" else (null <= x)).sum()) / (len(null) + 1))


def view(A: np.ndarray, R: np.ndarray, sig: np.ndarray) -> dict:
    """A [traits, arm + its random directions]; R the reference's change; sig the reference's traits."""
    r = np.array([np.corrcoef(A[:, c], R)[0, 1] for c in range(A.shape[1])])
    ag = np.array([(np.sign(A[sig, c]) == np.sign(R[sig])).mean() for c in range(A.shape[1])])
    o = {"r": float(r[0]), "r_random": r[1:].round(4).tolist(), "same_sign": float(ag[0]),
         "same_sign_random": ag[1:].round(4).tolist()}
    if A.shape[1] > 1:
        own = PS.perm_fdr(A) <= 0.05 + 1e-9
        o.update({"p_r": rank_p(r[0], r[1:]), "p_same_sign": rank_p(ag[0], ag[1:]), "perm_fdr": int(own.sum()),
                  "overlap": int((own & sig).sum())})
    return o


def main():
    ref = json.load(open(sys.argv[1]))
    traits, T, e = trait_T(ref)
    nm = ref["names"]
    cols = [nm.index("set")] + [j for j, n in enumerate(nm) if n.startswith("random")]
    RA, RB = T[:, cols], beyond(T[:, cols], e)
    sig_a, sig_b = PS.perm_fdr(RA) <= 0.05 + 1e-9, PS.perm_fdr(RB) <= 0.05 + 1e-9
    X = np.c_[np.ones(len(e)), e]
    sl = np.linalg.lstsq(X, RA, rcond=None)[0][1]
    out = {"traits": traits, "endorsement": e.round(4).tolist(),
           "reference": {"T": RA[:, 0].round(4).tolist(), "T_beyond": RB[:, 0].round(4).tolist(),
                         "perm_fdr": int(sig_a.sum()), "perm_fdr_beyond": int(sig_b.sum()),
                         "slope": float(sl[0]), "slope_random": sl[1:].round(4).tolist(),
                         "leans_further": int((np.sign(e) * RA[:, 0] > 0).sum()),
                         "leans_further_random": [int((np.sign(e) * RA[:, c] > 0).sum()) for c in range(1, RA.shape[1])],
                         "q_beyond": dict(zip(traits, PS.perm_fdr(RB).round(4).tolist()))},
           "arms": {}}
    for d in sys.argv[3:]:
        rec = json.load(open(Path(d) / "persona.json"))
        lies = json.load(open(Path(d) / "lies.json")) if (Path(d) / "lies.json").exists() else None
        tr, A_, e2 = trait_T(rec)
        if len(rec["items"]) < len(ref["items"]) or tr != traits:
            print(f"{d}: {len(rec['items'])} items, skipped")
            continue
        names = rec["names"]
        for arm in rec["arms"]:
            j = names.index(arm["name"])
            rj = [k for k, n in enumerate(names) if n.startswith(arm["name"] + "/random")]
            A = A_[:, [j] + rj]
            o = {"alpha": float(arm["alpha"]), "kind": arm["kind"], "random": len(rj),
                 "raw": view(A, RA[:, 0], sig_a), "beyond": view(beyond(A, e2), RB[:, 0], sig_b),
                 "T": A[:, 0].round(4).tolist(), "slope": float(np.linalg.lstsq(X, A, rcond=None)[0][1, 0])}
            if rj:
                o["slope_random"] = np.linalg.lstsq(X, A, rcond=None)[0][1, 1:].round(4).tolist()
                for key, SET in (("deception", PS.DECEPTION), ("dark", PS.DARK)):
                    ix = [traits.index(t) for t in traits if t in SET]
                    D = (A[ix] * np.array([SET[traits[i]] for i in ix])[:, None]).mean(0)
                    o[key] = {"T": float(D[0]), "random": D[1:].round(4).tolist(), "p_less": rank_p(D[0], D[1:], "less")}
            if lies:
                lj = lies["names"].index(arm["name"]) - 1
                lr = [lies["names"].index(n) - 1 for n in lies["names"] if n.startswith(arm["name"] + "/random")]
                o["lies"] = {}
                for f, v in lies["families"].items():
                    x = np.array(v["steered"])
                    hit = ((x < 0) if f.endswith(" lie") else (x > 0)).mean(0)
                    o["lies"][f] = {"n": len(x), "share": float(hit[lj]), "random": hit[lr].round(4).tolist()}
                o["controls"] = {g: {"n": v["n"], "stay": v["stay"][lj], "random": [v["stay"][k] for k in lr]}
                                 for g, v in lies["controls"].items()}
            out["arms"][arm["name"]] = o
            b = o["beyond"]
            print(f"{arm['name']:32s} alpha {o['alpha']:6.3g}  raw r {o['raw']['r']:+.2f}  beyond confidence r "
                  f"{b['r']:+.2f} (random max {max(b['r_random'], default=float('nan')):+.2f}, p "
                  f"{b.get('p_r', float('nan')):.3f})  own FDR {b.get('perm_fdr')} overlap {b.get('overlap')}  "
                  f"slope {o['slope']:+.3f}")
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
