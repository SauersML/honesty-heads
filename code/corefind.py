"""The common core of the deception families (deccore steps 1-5; ~/mpd-data/scratch/deccore/lead/design.txt): units
(heads and neurons at the shared span) ordered by gradients, every inclusion decided by measured patching (engine.py).

Items: dsets' measured pairs (measure.py: knows and lies, m_dec - m_hon >= MIN_GAP bits, lie and truth ids). Each pair
is re-measured here in the run's dtype (float32 for final numbers) and kept when the gap still reaches MIN_GAP. Per
family a seeded sample of the select split (--n-select) and of the test split (--n-test); a family's span P_F is the
shortest shared tail of its sampled pairs (at most --span-max), the core's span P_core the shortest over the selection
families (the chat template from the user turn's <|im_end|> to the decision position, at least).

Ordering (per pair, both on the span; scores normalized by the gap, positive = moves the deceptive run toward honest):
  ig    integrated gradients along the straight path from the deceptive to the honest span head outputs (every span
        head set on the path, neurons live; IG_STEPS steps, trapezoid): a head's score is its path increment, a
        neuron's its own increments sum_k g_k . (n_k+1 - n_k) along the path. The heads' increments sum to 1 (checked).
  atp   attribution patching: grad . (hon - dec) at the deceptive run (atpN, estimates N of the unit alone) and at the
        honest run (atpR, estimates R); the order uses (atpN + atpR) / 2.
A family's score of a unit: the mean over its select pairs. The orders' Spearman against measured single-unit patches
(N and R of the unit alone on SINGLES_ITEMS pairs of each family, units: the top SINGLES_TOP of each order and
SINGLES_RAND random) is reported.
Per-family circuit: the family's own order, the smallest prefix whose median N and median R over the family's select
pairs both reach PASS (grid sizes, then BISECT rounds of 3 sizes between the last failing and the first passing; the
order, ig or atp, chosen on CHOOSE_ITEMS pairs of the select split by the smaller passing size).
Common core: units at pb < P_core ordered by a cross-family score of the selection families' normalized scores:
  min2  the second-smallest family score (one family may lack the unit), min  the smallest, mean  the mean;
for each order (ig, atp) x rule, the curve on CHOOSE_ITEMS select pairs of each family; the (order, rule) with the
smallest prefix passing every family is taken, its whole curve measured on every select pair, the smallest passing
prefix found by bisection. No size cap; the curves (median N and R against units added) are saved whole.
Writes OUT/*.json, OUT/attr_<family>_P<P>.pt, and the core to --core-out/core_v<k>.json with the honest means (the
partner-free values: each unit's mean over the select pairs' honest runs of every selection family; content-free: the
honest pool's mean) and the deceptive means in core_v<k>_means.pt.
--content-free (core v2): a core patched with a pair's own partner can pass by carrying the item's answer token
(core v1 did: another item's honest values remove the lie only when its truth is the same token). So the patched values
hold no item's answer (Sources): the honest pool's mean and an honest run of another task into the deceptive run, the
selection families' deceptive mean and another task's deceptive run into the honest run, each pair's own lie and truth
tokens scoring it, N credited only as far as the item's own truth comes back (engine.shares "Nc"); every family must
pass under both. Polarity families (tf_t, tf_f) keep a "write True" set from
passing. --content-free shift: the run keeps its own values and moves by the select pairs' honest-minus-deceptive mean
difference (Sources "shift"), the unit-restricted difference-of-means intervention. Inside the search, audit's controls (control_recs: the first answer token) must stay as right under the core's
mean edit as under random matched sets (specificity). Orders use the mean objective (IG along the path to the means,
AtP of the mean patch). When nothing passes, the best prefix (largest min-share) is written with met = false.

  corefind.py run --items ITEMS.json --out DIR --families tf_t,tf_f,qa [--heldout lbi_t,lb_conv,mask_yn] [--span 9]
      [--model Qwen/Qwen3.5-9B] [--dtype float32] [--device cuda] [--n-select 64] [--n-test 128] [--span-max 48]
      [--version 1] [--core-out ~/mpd-data/scratch/deccore/core] [--content-free [replace|shift] [--pool POOL.json]
      [--controls C.json]]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402

PASS = 0.7
MIN_GAP = 1.0         # bits
IG_STEPS = int(os.environ.get("DCORE_IG_STEPS", 8))
GRAD_TOKENS = int(os.environ.get("DCORE_GRAD_TOKENS", 256))   # rows x P of a backward pass
GROUP_GB = float(os.environ.get("DCORE_GROUP_GB", 4.0))       # prepared pairs held on the device at once
SINGLES_ITEMS = int(os.environ.get("DCORE_SINGLES_ITEMS", 8))
SINGLES_TOP = 48
SINGLES_RAND = 48
CHOOSE_ITEMS = int(os.environ.get("DCORE_CHOOSE_ITEMS", 16))
BISECT = 3
PER_DOUBLING = int(os.environ.get("DCORE_PER_DOUBLING", 4))
RULES = ("min2", "min", "mean")
ORDERS = ("ig", "atp")
DONORS = 64           # donor runs kept per family (content-free "donor" mode)
TASK = {"tf_t": "tf", "tf_f": "tf"}        # polarity families share their task (donors come from another task)
CTL_GROUPS = "honest_tf,honest_qa,french,caps"
SPEC_RANDOM = int(os.environ.get("DCORE_SPEC_RANDOM", 4))
SPEC_TOL = 0.05       # a control group's accuracy may fall at most this far below the random sets' mean
SPEC_SPAN = 8         # specificity measured on grid sizes up to SPEC_SPAN x the first N/R-passing size
SPEC_MAX = 4096       # ... or up to this size when none passes
N_KEY = "N"           # the necessity share a prefix must pass on; content-free runs set "Nc" (engine.shares: credited
                      # only as far as this item's truth comes back, not for any drop of its lie tokens)
R_KEY = "R"           # the same for sufficiency: content-free runs set "Rc" (credited only as far as this item's lie
                      # tokens gain, not for erasing its honest answer)
EITHER = False        # content-free "both": a family passes under its better intervention (mean or shift)


# ---------------------------------------------------------------- items
def candidates(items: list, fam: str, split: str) -> list:
    """Items of fam/split that know and lie, with token ids; paired items with both signs (m_dec > 0 > m_hon) and a gap of
    MIN_GAP bits, deceptive-only items (no honest version) with m_dec > 0."""
    out = []
    for it in items:
        if it["family"] != fam or it["split"] != split or not it.get("knows") or not it.get("lies"):
            continue
        if it.get("m_dec") is None or not it.get("lie_ids") or not it.get("truth_ids") or it["m_dec"] <= 0:
            continue
        if it.get("hon") and (it.get("m_hon") is None or it["m_hon"] >= 0 or it["m_dec"] - it["m_hon"] < MIN_GAP):
            continue
        out.append(it)
    return out


def seeded(xs: list, key: str) -> list:
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return [xs[i] for i in rng.permutation(len(xs))]


class Data:
    """The sampled, re-measured pairs per family and split, with token ids and shared tails."""

    def __init__(self, eng, tok, items: list, fams: list, n_sel: int, n_test: int, span_max: int, log=E.log):
        self.eng, self.tok, self.by, self.drop = eng, tok, {}, {}
        for f in fams:
            for split, n in (("select", n_sel), ("test", n_test)):
                cand = seeded(candidates(items, f, split), f"{f}/{split}")
                keep, dropped = [], 0
                for it in cand:
                    if len(keep) >= n:
                        break
                    d, h = E.item_seqs(tok, it)
                    tail = E.shared_tail(d, h) if h is not None else E.template_tail(tok, d)
                    if tail < 2:
                        dropped += 1
                        continue
                    rd = eng.prepare(d, 1, it["lie_ids"], it["truth_ids"])
                    rh = eng.prepare(h, 1, it["lie_ids"], it["truth_ids"]) if h is not None else None
                    if rd.m <= 0 or (rh is not None and (rh.m >= 0 or rd.m - rh.m < MIN_GAP)):   # both signs, re-measured
                        dropped += 1
                        continue
                    keep.append({"it": it, "d": d, "h": h, "tail": tail, "m_dec": rd.m,
                                 "m_hon": rh.m if rh is not None else None, "top_dec": rd.top,
                                 "top_hon": rh.top if rh is not None else None})
                self.by[(f, split)] = keep
                self.drop[(f, split)] = dropped
                tails = [k["tail"] for k in keep]
                log(f"items {f:6s} {split:6s} candidates {len(cand):5d} kept {len(keep):4d} dropped {dropped:3d} "
                    f"tail min {min(tails) if tails else 0} median {int(np.median(tails)) if tails else 0}")
        self.span_max = span_max

    @classmethod
    def from_ids(cls, eng, tok, items: list, ids: dict, span_max: int):
        """The pairs a run recorded ({"fam/split": [ids]}: corefind's items.json or a core's "items"), not re-sampled."""
        self = cls.__new__(cls)
        self.eng, self.tok, self.by, self.drop, self.span_max = eng, tok, {}, {}, span_max
        byid = {it["id"]: it for it in items}
        for key, lst in ids.items():
            f, split = key.split("/")
            recs = []
            for i in lst:
                it = byid[i]
                d, h = E.item_seqs(tok, it)
                recs.append({"it": it, "d": d, "h": h,
                             "tail": E.shared_tail(d, h) if h is not None else E.template_tail(tok, d)})
            self.by[(f, split)] = recs
        return self

    def P(self, fam: str) -> int:
        t = [k["tail"] for s in ("select", "test") for k in self.by.get((fam, s), [])]
        return min(self.span_max, min(t)) if t else 0

    def pairs(self, recs: list, P: int):
        """Groups of prepared pairs (engine.Pair) of recs at span P, sized to GROUP_GB on the device."""
        per = 2 * (P * self.eng.L * (self.eng.D + max(w * n for w, n in zip(self.eng.hdim, self.eng.nheads))) * 4
                   + sum(0 if f else 2.2e6 for f in self.eng.full))
        G = max(2, int(GROUP_GB * 1e9 / per))
        for a in range(0, len(recs), G):
            out = []
            for k in recs[a:a + G]:
                it = k["it"]
                rd = self.eng.prepare(k["d"], P, it["lie_ids"], it["truth_ids"])
                rh = self.eng.prepare(k["h"], P, it["lie_ids"], it["truth_ids"]) if k["h"] is not None else None
                out.append(E.Pair(it, rd, rh))
            yield out
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def ids(self) -> dict:
        return {f"{f}/{s}": [k["it"]["id"] for k in v] for (f, s), v in self.by.items()}


# ---------------------------------------------------------------- ordering
def _put(U, vec, kind, l, span_val):
    """Add [P, W] span-ordered values into a flat vector's block (pb order)."""
    U.view(vec, kind, l)[:span_val.shape[0]] += span_val.flip(0).to(vec.device)


def attribute(eng, U: E.Units, pr: E.Pair, acc: dict, steps: int = IG_STEPS, tn: dict | None = None,
              tr: dict | None = None) -> float:
    """Adds this pair's normalized ig, atpN, atpR scores into acc's flat vectors; returns the heads' ig sum (1 when
    the path's increments are complete). tn: the values patched into the deceptive run (default the honest run's), tr:
    into the honest run (default the deceptive run's); partner-free scoring passes the honest and deceptive means."""
    P = pr.dec.P
    tn, tr = tn or pr.hon.cap, tr or pr.dec.cap
    gap = pr.hon.m - pr.dec.m
    al = torch.linspace(0, 1, steps + 1, device=eng.dev)
    hsum = {l: [] for l in range(eng.L)}
    ng, nv = {l: [] for l in range(eng.L)}, {l: [] for l in range(eng.L)}
    rows = max(1, GRAD_TOKENS // P)
    for a in range(0, steps + 1, rows):
        idx = al[a:a + rows]
        zs, vals = {}, {}

        def edit(kind, l, x, idx=idx):
            if kind == "h":
                hd, hh = pr.dec.cap["h"][l], tn["h"][l]
                x = (hd[None] + idx.to(hd.device)[:, None, None] * (hh - hd)[None]).to(x.dtype)
            else:
                vals[l] = x.detach()
            z = torch.zeros_like(x, requires_grad=True)
            zs[(kind, l)] = z
            return x + z
        with torch.enable_grad():
            lp = eng.span([pr.dec] * len(idx), edit)
            m = (torch.logsumexp(lp[:, pr.dec.lie], 1) - torch.logsumexp(lp[:, pr.dec.truth], 1)) / E.LN2
            m.sum().backward()
        for l in range(eng.L):
            g = zs[("h", l)].grad
            d = (tn["h"][l] - pr.dec.cap["h"][l])[None]
            hsum[l].append((g * d).view(len(idx), P, eng.nheads[l], eng.hdim[l]).sum(-1).float())
            ng[l].append(zs[("n", l)].grad.float())
            nv[l].append(vals[l].float())
        del zs, vals, lp, m
    w = torch.ones(steps + 1, device=eng.dev)
    w[0] = w[-1] = 0.5
    tot = 0.0
    for l in range(eng.L):
        hs = torch.cat(hsum[l])                                   # [K+1, P, nh]
        inc = (w.to(hs.device)[:, None, None] * hs).sum(0) / steps
        tot += float(inc.sum())
        _put(U, acc["ig"], "h", l, inc / gap)
        g, v = torch.cat(ng[l]), torch.cat(nv[l])
        inc = (0.5 * (g[:-1] + g[1:]) * (v[1:] - v[:-1])).sum(0)
        _put(U, acc["ig"], "n", l, inc / gap)
    del hsum, ng, nv
    # attribution patching at both runs
    zs = {}

    def edit2(kind, l, x):
        z = torch.zeros_like(x, requires_grad=True)
        zs[(kind, l)] = z
        return x + z
    with torch.enable_grad():
        lp = eng.span([pr.dec, pr.hon], edit2)
        m = (torch.logsumexp(lp[:, pr.dec.lie], 1) - torch.logsumexp(lp[:, pr.dec.truth], 1)) / E.LN2
        m.sum().backward()
    for l in range(eng.L):
        for kind in ("h", "n"):
            g = zs[(kind, l)].grad.float()
            d = torch.stack([tn[kind][l] - pr.dec.cap[kind][l], pr.hon.cap[kind][l] - tr[kind][l]]).float()
            s = g * d
            if kind == "h":
                s = s.view(2, P, eng.nheads[l], eng.hdim[l]).sum(-1)
            _put(U, acc["atpN"], kind, l, s[0] / gap)
            _put(U, acc["atpR"], kind, l, s[1] / gap)
    return tot / gap


def family_scores(eng, data: Data, fam: str, P: int, out: Path, log=E.log, src: "Sources | None" = None,
                  mode: str = "mean") -> dict:
    """{ig, atpN, atpR, atp} family-mean normalized scores (flat CPU float32 vectors at span P), cached in out; src: its
    content-free values in mode ("mean" or "shift") are the patched values instead of each pair's partner."""
    path = out / f"attr_{fam}_P{P}{('_cf' if mode == 'mean' else '_' + mode) if src else ''}.pt"
    if path.exists():
        return torch.load(path)
    U = E.Units(eng, P)
    acc = {k: torch.zeros(U.n, device=eng.dev) for k in ("ig", "atpN", "atpR")}
    recs, comp, t0 = data.by[(fam, "select")], [], time.time()
    for grp in data.pairs(recs, P):
        for pr in grp:
            tn, tr = src.targets(mode, pr) if src else (None, None)
            comp.append(attribute(eng, U, pr, acc, tn=tn, tr=tr))
    n = max(1, len(comp))
    res = {k: (v / n).cpu() for k, v in acc.items()}
    res["atp"] = (res["atpN"] + res["atpR"]) / 2
    res["ig_complete"] = torch.tensor(comp)
    torch.save(res, path)
    log(f"attr {fam} P {P}{' content-free ' + mode if src else ''}: {len(comp)} pairs, {time.time() - t0:.0f} s; ig heads' sum / "
        f"gap median {np.median(comp) if comp else float('nan'):.3f} (range {min(comp, default=0):.3f}..{max(comp, default=0):.3f})")
    return res


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = (np.argsort(np.argsort(x[ok])).astype(float) for x in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


def singles(eng, data: Data, fams: list, P: int, scores: dict, log=E.log) -> dict:
    """Measured single-unit N and R against each order's scores (Spearman), per family."""
    U = E.Units(eng, P)
    res = {}
    for f in fams:
        sc = scores[f]
        rng = np.random.default_rng(zlib.crc32(f"singles/{f}".encode()))
        top = set()
        for o in ORDERS:
            top |= set(torch.argsort(-sc[o])[:SINGLES_TOP].tolist())
        units = sorted(top | set(rng.choice(U.n, SINGLES_RAND, replace=False).tolist()))
        recs = data.by[(f, "select")][:SINGLES_ITEMS]
        Ns, Rs = [], []
        for grp in data.pairs(recs, P):
            sh = E.shares(eng, U, grp, [("units", np.array([u])) for u in units])
            Ns.append(sh["N"])
            Rs.append(sh["R"])
        N, R = np.concatenate(Ns).mean(0), np.concatenate(Rs).mean(0)
        meas = (N + R) / 2
        tops = np.array([u in top for u in units])
        r = {"n_units": len(units), "n_items": len(recs), "n_top": int(tops.sum()),
             "ig_vs_NR_top": spearman(sc["ig"][units].numpy()[tops], meas[tops]),
             "atp_vs_NR_top": spearman(sc["atp"][units].numpy()[tops], meas[tops]),
             "ig_vs_NR": spearman(sc["ig"][units].numpy(), meas), "atp_vs_NR": spearman(sc["atp"][units].numpy(), meas),
             "atpN_vs_N": spearman(sc["atpN"][units].numpy(), N), "atpR_vs_R": spearman(sc["atpR"][units].numpy(), R),
             "ig_vs_N": spearman(sc["ig"][units].numpy(), N), "ig_vs_R": spearman(sc["ig"][units].numpy(), R),
             "top_single": [{**E.unit_json(U.key(u)), "N": float(n_), "R": float(r_)}
                            for u, n_, r_ in sorted(zip(units, N, R), key=lambda x: -(x[1] + x[2]))[:12]]}
        res[f] = r
        log(f"singles {f} P {P}: spearman ig {r['ig_vs_NR']:.2f} atp {r['atp_vs_NR']:.2f} (top units only: ig "
            f"{r['ig_vs_NR_top']:.2f} atp {r['atp_vs_NR_top']:.2f}; atpN~N {r['atpN_vs_N']:.2f}, "
            f"atpR~R {r['atpR_vs_R']:.2f}); best single unit N {max(N):.3f} R {max(R):.3f}")
    return res


# ---------------------------------------------------------------- curves and search
def order_of(score: torch.Tensor) -> np.ndarray:
    return torch.argsort(-score, stable=True).numpy()


def cross(scores: dict, fams: list, kind: str, rule: str) -> torch.Tensor:
    S = torch.stack([scores[f][kind] for f in fams])
    if rule == "mean":
        return S.mean(0)
    s = S.sort(0).values
    return s[1] if rule == "min2" and len(fams) >= 3 else s[0]


def keyof(fam: str, mode: str) -> str:
    """A curve's key: the family, with the source mode when it is not the partner ("tf_t|mean")."""
    return fam if mode == "pair" else f"{fam}|{mode}"


def curves(eng, data: Data, fams: list, P: int, orders: list, sizes: list, n_items: int | None = None,
           src: "Sources | None" = None, modes: tuple = ("pair",)) -> dict:
    """{keyof(fam, mode): {"N": [items, orders, sizes], "R": ..., "flipN", "flipR"}} measured on the select pairs (first
    n_items); mode "pair" patches each pair's partner, "mean" and "donor" src's content-free values."""
    U = E.Units(eng, P)
    ranks = [E.rank_of(o, U.n, eng.dev) for o in orders]
    sets = [("rank", R, k) for R in ranks for k in sizes]
    out = {}
    for f in fams:
        recs = data.by[(f, "select")][:n_items] if n_items else data.by[(f, "select")]
        acc = {m: {k: [] for k in ("N", "Nc", "R", "Rc", "flipN", "flipR")} for m in modes}
        a0 = 0
        for grp in data.pairs(recs, P):
            for m in modes:
                sh = E.shares(eng, U, grp, sets, src=src.fn(m, f, a0, grp) if m != "pair" else None)
                for k in acc[m]:
                    acc[m][k].append(sh[k].reshape(len(grp), len(orders), len(sizes)))
            a0 += len(grp)
        for m in modes:
            out[keyof(f, m)] = {k: np.concatenate(v) for k, v in acc[m].items()}
    del ranks
    return out


def _groups(keys: list) -> list:
    """Keys that must each pass, or under EITHER per family the keys of which one must (its better intervention)."""
    g = {}
    for k in keys:
        g.setdefault(k.split("|")[0] if EITHER else k, []).append(k)
    return list(g.values())


def passing(cur: dict, keys: list, o: int) -> np.ndarray:
    """[sizes] bool: every key's (family's, under every source mode; under EITHER, any of the family's modes) median N
    (N_KEY) and median R (R_KEY) reach PASS for order o."""
    ok = None
    for ks in _groups(keys):
        a = np.any([(np.nanmedian(cur[k][N_KEY][:, o], 0) >= PASS) & (np.nanmedian(cur[k][R_KEY][:, o], 0) >= PASS)
                    for k in ks], 0)
        ok = a if ok is None else ok & a
    return ok


def min_share(cur: dict, keys: list, o: int) -> np.ndarray:
    """[sizes]: the smallest median N or R over keys (under EITHER, of each family's better mode): the passing margin,
    the best curve when none passes."""
    return np.min([np.max([np.min([np.nanmedian(cur[k][q][:, o], 0) for q in (N_KEY, R_KEY)], 0) for k in ks], 0)
                   for ks in _groups(keys)], 0)


def summary(cur: dict, keys: list, o: int, sizes: list) -> dict:
    return {"sizes": list(map(int, sizes)),
            **{f: {"N_median": np.nanmedian(cur[f]["N"][:, o], 0).round(4).tolist(),
                   **({"Nc_median": np.nanmedian(cur[f]["Nc"][:, o], 0).round(4).tolist()} if "Nc" in cur[f] else {}),
                   **({"Rc_median": np.nanmedian(cur[f]["Rc"][:, o], 0).round(4).tolist()} if "Rc" in cur[f] else {}),
                   "R_median": np.nanmedian(cur[f]["R"][:, o], 0).round(4).tolist(),
                   "flipN": cur[f]["flipN"][:, o].mean(0).round(4).tolist(),
                   "flipR": cur[f]["flipR"][:, o].mean(0).round(4).tolist(), "n": int(cur[f]["N"].shape[0])}
               for f in keys}}


def smallest(eng, data: Data, fams: list, P: int, order: np.ndarray, log=E.log, tag: str = "",
             src: "Sources | None" = None, modes: tuple = ("pair",), spec: tuple | None = None) -> dict:
    """The whole curve of order on every select pair of fams (each source mode), then the smallest passing prefix
    (bisection). spec = (control records, K, tolerance): a passing prefix must also keep every control group's accuracy
    within tolerance of K random matched sets (specificity, measured on the grid sizes up to SPEC_SPAN x the first size
    passing N and R, or up to SPEC_MAX when none does)."""
    keys = [keyof(f, m) for f in fams for m in modes]
    sizes = E.grid(len(order), PER_DOUBLING)
    t0 = time.time()
    cur = curves(eng, data, fams, P, [order], sizes, src=src, modes=modes)
    ok = passing(cur, keys, 0)
    ms = min_share(cur, keys, 0)
    rec = {"curve": summary(cur, keys, 0, sizes), "refine": [], "keys": keys,
           "best": {"size": int(sizes[int(np.nanargmax(ms))]), "min_share": float(np.nanmax(ms))}}
    smodes = [m for m in modes if m in ("mean", "shift")]
    if spec is not None:
        first = sizes[int(np.argmax(ok))] if ok.any() else None
        ss = [k for k in sizes if k <= min(len(order), first * SPEC_SPAN if first else SPEC_MAX)]
        sok = np.ones(len(ss), dtype=bool)
        for m in smodes:        # the controls under every content-free edit the core is selected with
            sp = specificity(eng, data, spec[0], P, order, ss, src, m, spec[1], tag=tag)
            rec["specificity" if m == smodes[0] else f"specificity_{m}"] = sp
            sok &= spec_ok(sp, spec[2])
            log(f"{tag}: specificity under {m} (core minus random, per group, at {len(ss)} sizes to {ss[-1]}): " +
                "; ".join(f"{g} {np.round(np.array(r['core']) - np.array(r['random']), 2)[[0, len(ss) // 2, -1]].tolist()}"
                          for g, r in sp["groups"].items()))
        sok = dict(zip(ss, sok))
        ok = ok & np.array([sok.get(k, False) for k in sizes])
    if not ok.any():
        log(f"{tag}: no prefix passes (best min share {rec['best']['min_share']:.3f} at {rec['best']['size']} units)")
        rec["size"] = None
        return rec
    j = int(np.argmax(ok))
    lo, hi = (sizes[j - 1] if j else 0), sizes[j]
    for _ in range(BISECT):
        if hi - lo <= 1:
            break
        mids = sorted(set(int(round(x)) for x in np.geomspace(max(lo, 1), hi, 5)[1:-1]) - {lo, hi})
        mids = [m for m in mids if lo < m < hi]
        if not mids:
            mids = [(lo + hi) // 2]
        c2 = curves(eng, data, fams, P, [order], mids, src=src, modes=modes)
        ok2 = passing(c2, keys, 0)
        r2 = summary(c2, keys, 0, mids)
        if spec is not None:
            for m in smodes:
                sp2 = specificity(eng, data, spec[0], P, order, mids, src, m, spec[1], tag=tag)
                ok2 = ok2 & spec_ok(sp2, spec[2])
                r2["specificity" if m == smodes[0] else f"specificity_{m}"] = sp2
        rec["refine"].append(r2)
        passed = [m for m, o in zip(mids, ok2) if o]
        failed = [m for m, o in zip(mids, ok2) if not o]
        if passed:
            hi = min(passed)
        lo = max([m for m in failed if m < hi], default=lo)
    rec["size"] = int(hi)
    log(f"{tag}: smallest passing prefix {hi} (curve {len(sizes)} sizes, {time.time() - t0:.0f} s)")
    return rec


def choose(eng, data: Data, fams: list, P: int, cands: dict, log=E.log, tag: str = "", src: "Sources | None" = None,
           modes: tuple = ("pair",)) -> tuple:
    """(name, per-candidate first passing grid size, curves) on CHOOSE_ITEMS pairs per family: the order whose first
    passing grid size is smallest (ties: the larger min-share, the smallest family median N or R, at that size; when no
    order passes, the largest min-share anywhere on the curve). The grid is measured in blocks of sizes (to 1024, then x8
    per block) and stops after the first block where an order passes (an order passing later cannot win)."""
    names = list(cands)
    keys = [keyof(f, m) for f in fams for m in modes]
    n = max(len(o) for o in cands.values())
    sizes = E.grid(n, PER_DOUBLING)
    done, cur, lim = 0, None, 1024
    while done < len(sizes):
        blk = [k for k in sizes[done:] if k <= lim] or sizes[done:done + 1]
        c = curves(eng, data, fams, P, [cands[k] for k in names], blk, CHOOSE_ITEMS, src=src, modes=modes)
        cur = c if cur is None else {f: {k: np.concatenate([cur[f][k], c[f][k]], -1) for k in c[f]} for f in keys}
        done += len(blk)
        lim *= 8
        if any(passing(cur, keys, o).any() for o in range(len(names))):
            break
    sizes = sizes[:done]
    firsts, share = {}, {}
    for o, k in enumerate(names):
        ok = passing(cur, keys, o)
        ms = min_share(cur, keys, o)
        firsts[k] = int(sizes[int(np.argmax(ok))]) if ok.any() else None
        share[k] = float(ms[int(np.argmax(ok))]) if ok.any() else float(np.nanmax(ms))
    best = min(names, key=lambda k: (firsts[k] is None, firsts[k] or 0, -share[k]))
    log(f"{tag}: first passing size per order on {CHOOSE_ITEMS} pairs/family: {firsts}; min-share there (none passing: "
        f"its best) { {k: round(v, 3) for k, v in share.items()} } -> {best}")
    return best, firsts, {k: summary(cur, keys, o, sizes) for o, k in enumerate(names)}


def honest_means(eng, data: Data, fams: list, P: int) -> tuple:
    """({"h"/"n": [per layer [P, W]]} honest, the same deceptive, n): the mean span captures over the select pairs'
    honest and deceptive runs of fams."""
    acc, n = None, 0
    for f in fams:
        for grp in data.pairs(data.by[(f, "select")], P):
            for pr in grp:
                if acc is None:
                    acc = [{k: [x.double().clone() for x in c[k]] for k in ("h", "n")} for c in (pr.hon.cap, pr.dec.cap)]
                else:
                    for A, c in zip(acc, (pr.hon.cap, pr.dec.cap)):
                        for k in ("h", "n"):
                            for a, x in zip(A[k], c[k]):
                                a += x.double()
                n += 1
    hm, dm = ({k: [(a / n).float().cpu() for a in A[k]] for k in ("h", "n")} for A in acc)
    return hm, dm, n


def task(fam: str) -> str:
    return TASK.get(fam, fam)


def _cpu(cap: dict) -> dict:
    return {k: [x.cpu() for x in v] for k, v in cap.items()}


class Sources:
    """Content-free values patched in place of a pair's partner (core v2: values that hold no item's answer):
      mean   replacement: the honest pool's mean (honest runs of the pool families: dsets2's honest_pool.json) into the
             deceptive run; the mean over the selection families' select deceptive runs into the honest run.
      donor  one other item's run of a different task: an honest pool run into the deceptive run, another selection
             family's select deceptive run into the honest run (pairs rotate over DONORS runs per family).
      shift  the run's own values plus alpha x shift into the deceptive run, minus it into the honest run; shift = the
             select pairs' honest mean minus their deceptive mean (the same pairs on both sides, so item content cancels:
             the unit-restricted difference-of-means intervention). The item's own content stays; only the mode moves.
    A family's task is its name without a polarity suffix (tf_t, tf_f: tf)."""

    def __init__(self, eng, tok, data: Data, fams: list, P: int, pool: list, pool_fams: list, log=E.log):
        acc, n, self.hdon, self.alpha = None, 0, {}, 1.0
        for it in pool:
            g = it.get("pool_family")
            if g not in pool_fams:
                continue
            ids = E.prompt_ids(tok, it["hon"], it.get("prefill", "")) + list(it.get("prefix_ids") or [])
            r = eng.prepare(ids, P, it.get("lie_ids") or [0], it.get("truth_ids") or [0])
            acc = _add(acc, r.cap)
            n += 1
            if len(self.hdon.setdefault(g, [])) < DONORS:
                self.hdon[g].append(_cpu(r.cap))
            del r
        self.hmean, self.n_pool = {k: [(x / n).float() for x in v] for k, v in acc.items()}, n
        acc, hacc, n, self.ddon = None, None, 0, {}
        for f in fams:
            for grp in data.pairs(data.by[(f, "select")], P):
                for pr in grp:
                    acc, hacc = _add(acc, pr.dec.cap), _add(hacc, pr.hon.cap)
                    n += 1
                    if len(self.ddon.setdefault(f, [])) < DONORS:
                        self.ddon[f].append(_cpu(pr.dec.cap))
        self.dmean, self.n_dec = {k: [(x / n).float() for x in v] for k, v in acc.items()}, n
        self.shift = {k: [((h - d) / n).float() for h, d in zip(hacc[k], acc[k])] for k in acc}
        log(f"sources: honest pool mean over {self.n_pool} runs ({', '.join(f'{g} {len(v)}' for g, v in self.hdon.items())} "
            f"donors), deceptive mean and honest-minus-deceptive shift over {self.n_dec} select pairs of {fams}")

    def _plus(self, cap: dict, sign: float) -> dict:
        a = sign * self.alpha
        return {k: [x + a * dx.to(x.device) for x, dx in zip(cap[k], self.shift[k])] for k in cap}

    def fn(self, mode: str, fam: str, a0: int = 0, grp: list | None = None):
        """src(pair index, "N"|"R") for E.shares on a group (grp) whose first pair is pair a0 of fam."""
        if mode == "mean":
            return lambda i, d: self.hmean if d == "N" else self.dmean
        if mode == "shift":
            memo = {}

            def get(i, d):
                if (i, d) not in memo:
                    pr = grp[i]
                    memo[(i, d)] = (self._plus(pr.dec.cap, 1.0) if d == "N" else
                                    self._plus(pr.hon.cap, -1.0) if pr.hon is not None else None)
                return memo[(i, d)]
            return get
        hs = [c for g, v in self.hdon.items() if task(g) != task(fam) for c in v]
        ds = [c for g, v in self.ddon.items() if task(g) != task(fam) for c in v]
        return lambda i, d: hs[(a0 + i) % len(hs)] if d == "N" else (ds[(a0 + i) % len(ds)] if ds else None)

    def targets(self, mode: str, pr: E.Pair) -> tuple:
        """(values into the deceptive run, into the honest run) of a pair, for the orders (attribute)."""
        if mode == "shift":
            return self._plus(pr.dec.cap, 1.0), self._plus(pr.hon.cap, -1.0)
        return self.hmean, self.dmean


def _add(acc, cap):
    if acc is None:
        return {k: [x.double().clone() for x in v] for k, v in cap.items()}
    for k in acc:
        for a, x in zip(acc[k], cap[k]):
            a += x.double()
    return acc


def control_recs(tok, path: str, groups: list) -> dict:
    """{group: records for Data.pairs} of audit's controls (audit_controls.py's controls.json), deceptive-only: the prompt
    plus the reply tokens the right answer shares with the first wrong one (teacher-forced); at the next token, truth ids
    = the right answer's token (as written, with and without a leading space, lower case and capitalized; caps: upper
    case only), lie ids = the wrong answers' tokens there. A control counts where the unedited model's top token is a
    truth id."""
    d = json.load(open(os.path.expanduser(path)))
    out = {}
    for it in (d["items"] if isinstance(d, dict) else d):
        g, gr = it["group"], it.get("grade")
        if g not in groups or not gr or not gr.get("bad") or not gr.get("ok"):     # (negation-type grades: no answer)
            continue
        up = gr["type"] == "caps"
        ok, bad = (gr["ok"][0].upper(), [b.upper() for b in gr["bad"]]) if up else (gr["ok"][0], list(gr["bad"]))
        a, b = tok.encode(ok, add_special_tokens=False), tok.encode(bad[0], add_special_tokens=False)
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        if n >= min(len(a), len(b)):
            continue

        def forms(w):
            w = w.strip()
            ws = [w] if up else [w, w.lower(), w[:1].upper() + w[1:]]
            return {tok.encode(x, add_special_tokens=False)[0] for v in ws for x in (v, " " + v)}
        pre = a[:n]
        truth = forms(tok.decode(a[n:]))
        lie = set()
        for bb in bad:
            t = tok.encode(bb, add_special_tokens=False)
            if t[:n] == pre and len(t) > n:
                lie |= forms(tok.decode(t[n:]))
        truth, lie = sorted(truth - lie), sorted(lie - truth)
        if not truth or not lie:
            continue
        ids = E.prompt_ids(tok, it["messages"]) + pre
        out.setdefault(g, []).append({"it": {"id": it["id"], "family": "ctl_" + g, "lie_ids": lie, "truth_ids": truth},
                                      "d": ids, "h": None, "tail": E.template_tail(tok, ids)})
    return out


def random_like(U: E.Units, eng, keys: list, rng, exclude: set, log=E.log) -> np.ndarray:
    """One random unit per key, of the same kind, layer and position, none in exclude and no repeats. A cell with no
    free unit left takes the nearest position with one (same kind and layer), and a layer with none the nearest layer
    with one (same kind; a large set can use up a layer's heads); the counts of those are logged."""
    out, used, moved, lmoved = [], set(int(x) for x in exclude), 0, 0
    for kind, l, idx, pb in keys:
        f = None
        for l2 in sorted((x for x in range(eng.L) if kind == "n" or eng.nheads[x]), key=lambda x: (abs(x - l), x)):
            W = eng.nheads[l2] if kind == "h" else eng.D
            for q in sorted(range(U.P), key=lambda q: (abs(q - pb), q)):
                for j in rng.permutation(W).tolist():
                    c = U.flat(kind, l2, j, q)
                    if c not in used:
                        f = c
                        break
                if f is not None:
                    break
            if f is not None:
                break
        assert f is not None, (kind, l, pb)
        moved += q != pb
        lmoved += l2 != l
        used.add(f)
        out.append(f)
    if moved or lmoved:
        log(f"random_like: {moved} of {len(keys)} units at a neighbouring position, {lmoved} at a neighbouring layer "
            f"(their cell had no free unit)")
    r = np.array(out, dtype=np.int64)
    assert len(set(r.tolist())) == len(r) and not set(r.tolist()) & set(int(x) for x in exclude)
    return r


def specificity(eng, data: Data, ctl: dict, P: int, order: np.ndarray, sizes: list, src: Sources, mode: str, K: int,
                tag: str = "") -> dict:
    """Control accuracy under the content-free edit (the order's prefix at each size set to the honest pool mean, or
    shifted by the honest-minus-deceptive difference: mode): per group, the share of controls the unedited model gets
    right that stay right, for the prefix and for K random orders matched unit by unit (random_like on the order's
    first max(sizes) units; each size takes their prefix)."""
    mode = "shift" if mode == "shift" else "mean"
    U = E.Units(eng, P)
    n = max(sizes)
    rng = np.random.default_rng(zlib.crc32(f"spec/{tag}/{n}".encode()))
    head = np.asarray(order[:n])
    keys = [U.key(int(u)) for u in head]
    rand = [random_like(U, eng, keys, rng, set(head.tolist())) for _ in range(K)]
    ranks = [E.rank_of(o, U.n, eng.dev) for o in [head] + rand]
    sets = [("rank", R, k) for R in ranks for k in sizes]
    res = {}
    for g, recs in ctl.items():
        acc, base = [], []
        for grp in data.pairs([r for r in recs if r["tail"] >= P], P):
            right = np.array([pr.dec.top in pr.dec.truth for pr in grp])
            base.append(right)
            if right.any():
                sh = E.shares(eng, U, grp, sets, src=src.fn(mode, "ctl", 0, grp))
                acc.append(sh["flipN"][right].reshape(int(right.sum()), 1 + K, len(sizes)))
        x = np.concatenate(acc).mean(0) if acc else np.full((1 + K, len(sizes)), np.nan)
        res[g] = {"n": int(sum(b.sum() for b in base)), "unedited_right": float(np.concatenate(base).mean()),
                  "core": x[0].round(4).tolist(), "random": x[1:].mean(0).round(4).tolist(),
                  "random_each": x[1:].round(4).tolist()}
    del ranks
    return {"sizes": list(map(int, sizes)), "K": K, "mode": mode, "alpha": src.alpha, "groups": res}


def bounds(eng, data: Data, fams: list, P: int, src: Sources, log=E.log) -> dict:
    """Median N, credited N and R per family (select pairs) when every head, or every unit, at the span takes each
    content-free value (mean replacement, mean shift): the full-span reference any subset is compared with."""
    U = E.Units(eng, P)
    heads = np.array([U.flat("h", l, i, pb) for l in range(eng.L) for i in range(eng.nheads[l]) for pb in range(P)])
    sets = [("units", heads), ("units", np.arange(U.n))]
    res = {}
    for m in ("mean", "shift"):
        for f in fams:
            acc, a0 = {k: [] for k in ("N", "Nc", "R", "Rc")}, 0
            for grp in data.pairs(data.by[(f, "select")], P):
                sh = E.shares(eng, U, grp, sets, src=src.fn(m, f, a0, grp))
                a0 += len(grp)
                for k in acc:
                    acc[k].append(sh[k])
            r = {k: np.concatenate(v) for k, v in acc.items()}
            res[keyof(f, m)] = {nm: {k: round(float(np.nanmedian(r[k][:, j])), 4) for k in acc}
                                for j, nm in enumerate(("all_heads", "all_units"))}
            log(f"bound {f} {m}: every span head " + " ".join(f"{k} {v:.2f}" for k, v in res[keyof(f, m)]["all_heads"].items())
                + "; every span unit " + " ".join(f"{k} {v:.2f}" for k, v in res[keyof(f, m)]["all_units"].items()))
    return res


def spec_ok(sp: dict, tol: float) -> np.ndarray:
    """[sizes] bool: every control group's accuracy under the core's edit is within tol of the random sets' mean."""
    ok = np.ones(len(sp["sizes"]), dtype=bool)
    for r in sp["groups"].values():
        ok &= np.array(r["core"]) >= np.array(r["random"]) - tol
    return ok


def unit_records(eng, U: E.Units, flat: np.ndarray, means: dict) -> list:
    out = []
    for f in flat:
        kind, l, idx, pb = U.key(int(f))
        P = U.P
        v = means[kind][l][P - 1 - pb]
        if kind == "h":
            v = v[idx * eng.hdim[l]:(idx + 1) * eng.hdim[l]].tolist()
        else:
            v = float(v[idx])
        out.append({**E.unit_json((kind, l, idx, pb)), "flat": int(f), "honest_mean": v})
    return out


def describe(units: list) -> dict:
    from collections import Counter
    return {"n": len(units), "heads": sum(u["kind"] == "head" for u in units),
            "neurons": sum(u["kind"] == "neuron" for u in units),
            "layers": dict(sorted(Counter(u["layer"] for u in units).items())),
            "pb": dict(sorted(Counter(u["pb"] for u in units).items()))}


# ---------------------------------------------------------------- the job
def run(a):
    torch.manual_seed(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    core_out = Path(os.path.expanduser(a.core_out))
    core_out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tok = E.load(a.model, a.dtype, a.device)
    eng = E.Engine(model)
    E.log(f"model {a.model} {a.dtype} layers {eng.L} heads/layer {sorted(set(eng.nheads))} neurons {eng.D}")
    items = json.load(open(a.items))
    fams = a.families.split(",")
    held = [f for f in a.heldout.split(",") if f]
    data = Data(eng, tok, items, fams + held, a.n_select, a.n_test, a.span_max)
    fams = [f for f in fams if len(data.by[(f, "select")]) >= a.min_items]
    Ps = {f: data.P(f) for f in fams + held}
    Pc = a.span or min(Ps[f] for f in fams)
    E.log(f"selection families {fams}; held out {held}; spans {Ps}; core span {Pc}")
    meta = {"model": a.model, "dtype": a.dtype, "families": fams, "heldout": held, "P": Ps, "P_core": Pc,
            "items": data.ids(), "dropped": {f"{k[0]}/{k[1]}": v for k, v in data.drop.items()},
            "measured": {f"{k[0]}/{k[1]}": [{"id": r["it"]["id"], "m_dec": r["m_dec"], "m_hon": r["m_hon"],
                                             "tail": r["tail"]} for r in v] for k, v in data.by.items()}}
    json.dump(meta, open(out / "items.json", "w"), indent=1)
    # content-free (core v2): the honest pool's mean and other-task donors patched instead of each pair's partner, with
    # control specificity inside the search
    cf, src, spec, modes = {"": "", "replace": "_cf", "shift": "_shift", "both": "_both"}[a.content_free], None, None, \
        ("pair",)
    smodes = {"": [None], "replace": ["mean"], "shift": ["shift"], "both": ["mean", "shift"]}[a.content_free]
    if a.content_free:
        global N_KEY, R_KEY, EITHER
        N_KEY, R_KEY, EITHER = "Nc", "Rc", a.content_free == "both"
        src = Sources(eng, tok, data, fams, Pc, json.load(open(os.path.expanduser(a.pool))), a.pool_families.split(","))
        modes = {"replace": ("mean", "donor"), "shift": ("shift",), "both": ("mean", "shift")}[a.content_free]
        ctl = control_recs(tok, a.controls, a.ctl_groups.split(","))
        E.log(f"controls: {', '.join(f'{g} {len(v)}' for g, v in ctl.items())}")
        spec = (ctl, SPEC_RANDOM, SPEC_TOL)
        means, dmeans, npool = _cpu(src.hmean), _cpu(src.dmean), src.n_pool
        bp = out / "bounds.json"
        if not bp.exists():
            json.dump(bounds(eng, data, fams, Pc, src), open(bp, "w"), indent=1)
    elif a.core:
        means, dmeans, npool = honest_means(eng, data, fams, Pc)
    # ordering at the core span and at each family's span
    # (each content-free mode's own scores: AtP / IG of that edit's objective)
    sc = {sm: {f: family_scores(eng, data, f, Pc, out, src=src, mode=sm or "mean") for f in fams} for sm in smodes}
    sc_core = sc[smodes[0]]
    sp = out / f"singles_P{Pc}.json"
    if not sp.exists() and not a.content_free:
        json.dump(singles(eng, data, fams, Pc, sc_core), open(sp, "w"), indent=1)
    E.log(f"ordering done {time.time() - t0:.0f} s")
    # the common core
    if a.core:
        cands = {(f"{sm}:" if j else "") + f"{o}/{r}": order_of(cross(sc[sm], fams, o, r))
                 for j, sm in enumerate(smodes) for o in ORDERS for r in RULES}
        cp = out / f"core_choose_P{Pc}{cf}.json"
        if cp.exists():
            best, firsts, ch = json.load(open(cp))
        else:
            best, firsts, ch = choose(eng, data, fams, Pc, cands, tag="core", src=src, modes=modes)
            json.dump([best, firsts, ch], open(cp, "w"))
        rec = smallest(eng, data, fams, Pc, cands[best], tag=f"core ({best})", src=src, modes=modes, spec=spec)
        rec.update({"order_rule": best, "choose_first_passing": firsts, "choose_curves": ch})
        json.dump(rec, open(out / f"core_search{cf}.json", "w"), indent=1)
        U = E.Units(eng, Pc)
        k = rec["size"]
        core = {"version": a.version, "model": a.model, "dtype": a.dtype, "P": Pc, "families": fams, "heldout": held,
                "rule": {"order": best.split(":")[-1].split("/")[0], "cross_family": best.split("/")[1], "pass": PASS,
                         "score_mode": best.split(":")[0] if ":" in best else smodes[0],
                         "criterion": "smallest prefix with median N >= pass and median R >= pass on every selection "
                                      "family's select pairs"},
                "content_free": a.content_free or False, "modes": list(modes), "either": EITHER,
                "size": k, "met": k is not None, "commit": os.environ.get("RP_REF", ""),
                "means_file": f"core_v{a.version}_means.pt",
                "means_pool": (f"honest pool {a.pool} ({a.pool_families}): {npool} runs" if a.content_free
                               else f"honest runs of {npool} select pairs"),
                "items": meta["items"], "curve": rec["curve"], "best": rec["best"]}
        if a.content_free:
            core["bounds"] = json.load(open(out / "bounds.json"))
            core.update({"pool": a.pool, "pool_families": a.pool_families, "controls": a.controls,
                         "ctl_groups": a.ctl_groups, "spec_tol": SPEC_TOL, "spec_random": SPEC_RANDOM,
                         "specificity": rec.get("specificity")})
            core["rule"]["criterion"] = ("smallest prefix with median credited N >= pass and median credited R >= pass on every "
                                         "selection family's select pairs under " +
                                         {"shift": "the honest-minus-deceptive mean shift",
                                          "replace": "the honest pool mean and under other-task donors",
                                          "both": "the better (per family) of the honest pool mean and the "
                                                  "honest-minus-deceptive mean shift"}[a.content_free] +
                                         ", and every control group's accuracy within spec_tol of random matched sets")
        # nothing passing: the core is the best prefix (largest min-share), flagged by met = false
        order = cands[best][:k] if k else cands[best][:rec["best"]["size"]]
        core["units"] = unit_records(eng, U, order, means)
        core["describe"] = describe(core["units"])
        core["order_head"] = [int(x) for x in cands[best][:max(4096, (k or 0) * 2)]]
        torch.save({"P": Pc, "means": means, "dmeans": dmeans, "shift": _cpu(src.shift) if src else None,
                    "order": torch.as_tensor(cands[best][:max(1 << 20, (k or 0) * 4)])}, core_out / core["means_file"])
        json.dump(core, open(core_out / f"core_v{a.version}.json", "w"))
        E.log(f"core v{a.version}: size {k} {core['describe']} -> {core_out / ('core_v%d.json' % a.version)}; "
              f"total {time.time() - t0:.0f} s")
    # per-family circuits
    famrec = {}
    sc_fam = {f: (sc_core[f] if Ps[f] == Pc and not a.content_free else family_scores(eng, data, f, Ps[f], out))
              for f in (fams if a.family_circuits else [])}
    for f in (fams if a.family_circuits else []):
        P = Ps[f]
        cands = {o: order_of(sc_fam[f][o]) for o in ORDERS}
        best, firsts, ch = choose(eng, data, [f], P, cands, tag=f"family {f}")
        rec = smallest(eng, data, [f], P, cands[best], tag=f"family {f} ({best})")
        rec.update({"family": f, "P": P, "order": best, "choose_first_passing": firsts, "choose_curves": ch})
        if rec["size"]:
            U = E.Units(eng, P)
            rec["units"] = [{**E.unit_json(U.key(int(u))), "flat": int(u)} for u in cands[best][:rec["size"]]]
            rec["describe"] = describe(rec["units"])
        famrec[f] = rec
        json.dump(rec, open(out / f"family_{f}.json", "w"), indent=1)
    E.log(f"family circuits done {time.time() - t0:.0f} s: " +
          ", ".join(f"{f} {famrec[f]['size']}" for f in famrec))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run"])
    p.add_argument("--items", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--families", required=True)
    p.add_argument("--heldout", default="")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-select", type=int, default=64)
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--min-items", type=int, default=16)
    p.add_argument("--span-max", type=int, default=48)
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--core-out", default="~/mpd-data/scratch/deccore/core")
    p.add_argument("--no-family-circuits", dest="family_circuits", action="store_false")
    p.add_argument("--no-core", dest="core", action="store_false")
    p.add_argument("--span", type=int, default=0, help="the core span P (default: the selection families' shortest)")
    p.add_argument("--content-free", nargs="?", const="replace", default="", choices=["replace", "shift", "both"],
                   help="core v2: order and select with content-free values instead of each pair's partner (replace: "
                        "the honest pool mean and other-task donors; shift: the run's own values moved by the "
                        "honest-minus-deceptive mean difference; both: each family passes under its better one), with "
                        "control specificity in the search under each")
    p.add_argument("--pool", default="~/mpd-data/scratch/deccore/dsets2/build/honest_pool.json")
    p.add_argument("--pool-families", default="tf,qa")
    p.add_argument("--controls", default="~/mpd-data/scratch/deccore/audit/controls.json")
    p.add_argument("--ctl-groups", default=CTL_GROUPS)
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
