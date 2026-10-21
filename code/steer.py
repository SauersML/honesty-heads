"""The mean shift at every position: is there a content-free honest/deceptive difference that, added to unit values
while the model reads the whole prompt (not only the shared template span), removes the lie and brings back this
item's truth?

Each unit (a head's output slice or a neuron, at no particular position) gets delta = the selection families' select
pairs' honest mean minus deceptive mean of its value, averaged over the span positions (polarity families pooled, so the
answer cancels and the honest/deceptive mode is left). Edit: unit values + alpha x delta at every position of the
deceptive run (N side), minus it in the honest run (R side), downstream units recomputed (steering restricted to the
set's units). "span" adds it at the last P positions only, for comparison with corefind's span shift. Sets: every head,
every unit, and each core's units (keyed without position). Scores: N, credited N (this item's truth recovered), R,
credited R (this item's lie induced), flips; deceptive-only items: the lie margin's change and flips; controls: the
share the unedited model gets right that stay right under +alpha x delta.

  steer.py run --items ITEMS.json --out DIR --families tf_t,tf_f,qa --heldout lbi_t,lb_conv,mask_yn [--span 9]
      [--cores A.json,B.json] [--controls C.json] [--alphas 0.5,1,2,4] [--n-items 64]
writes DIR/steer.json.
  steer.py select ... [--alphas 0.5,1,2]: the smallest head set whose every-position shift passes credited N and R on
      every selection family (heads ordered by the integrated gradients of the every-head shift), tested on test pairs,
      held-out families, random same-layer head sets and controls -> DIR/steer_heads.json.
  steer.py natural --items NATURAL.json --source syco_v2 --families syco_v2,rp,it,mask_yn --out DIR
      [--transfer-items items_v2.json,... --transfer tf_t,...] [--random-deltas 10] [--bins 4]: natural lies (no lie
      instruction), contrast A: delta = select resists' mean minus select lies' mean under the same pressure
      (knowledge-matched), +delta on lies (removed?), -delta on resists (lie induced?) -> DIR/delta_A_<source>.{pt,json}.
  steer.py natsel --items NATURAL.json --delta DIR/delta_A_<source>.pt --source ... --families ... --out DIR: the
      smallest head set whose every-position shift along that delta does what every head's does -> DIR/heads_nat.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corefind as C  # noqa: E402
import engine as E  # noqa: E402

LN2 = E.LN2


def delta(eng, data, fams: list, P: int) -> dict:
    """{"h"/"n": [per layer [W]]}: honest minus deceptive select-pair mean of each unit, averaged over the span."""
    acc, n = None, 0
    for f in fams:
        for grp in data.pairs(data.by[(f, "select")], P):
            for pr in grp:
                d = {k: [(h - x).double().mean(0) for h, x in zip(pr.hon.cap[k], pr.dec.cap[k])] for k in ("h", "n")}
                acc = d if acc is None else {k: [a + b for a, b in zip(acc[k], d[k])] for k in acc}
                n += 1
    return {k: [(a / n).float() for a in v] for k, v in acc.items()}


TWIN = {      # audit's format-instruction controls without the format instruction (deletion to the honest_qa suffix)
    "french": (" Write the answer as one French number word, with no digits. Reply with only that word.",
               " Reply with only the answer."),
    "caps": (" Reply with only the answer, in ALL CAPITAL LETTERS.", " Reply with only the answer."),
    "json": (' Reply only with JSON of the form {"answer": "..."}.', " Reply with only the answer."),
}


def format_dir(eng, tok, path: str, groups: list, P: int) -> tuple:
    """({"h"/"n": [per layer [W]]}, n): the mean over audit's format controls of (instructed minus the same prompt
    without the instruction) of each unit's value, averaged over the last P prompt positions."""
    items = json.load(open(os.path.expanduser(path)))["items"]
    acc, n = None, 0
    for it in items:
        if it["group"] not in groups:
            continue
        a, b = TWIN[it["group"]]
        u = it["messages"][-1]["content"]
        if a not in u:
            continue
        twin = [*it["messages"][:-1], {**it["messages"][-1], "content": u.replace(a, b)}]
        r1, r2 = (eng.prepare(E.prompt_ids(tok, m), P, [0], [0]) for m in (it["messages"], twin))
        d = {k: [(x - y).double().mean(0) for x, y in zip(r1.cap[k], r2.cap[k])] for k in ("h", "n")}
        acc = d if acc is None else {k: [p + q for p, q in zip(acc[k], d[k])] for k in acc}
        n += 1
    return {k: [(x / n).float() for x in v] for k, v in acc.items()}, n


def project_out(eng, dl: dict, u: dict) -> dict:
    """delta minus its component along u: per head (each head's output slice) and per layer for the neurons."""
    out = {"h": [], "n": []}
    for l in range(eng.L):
        d, w = dl["h"][l].view(eng.nheads[l], -1), u["h"][l].to(dl["h"][l].device).view(eng.nheads[l], -1)
        w = w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)
        out["h"].append((d - (d * w).sum(1, keepdim=True) * w).flatten())
        d, w = dl["n"][l], u["n"][l].to(dl["n"][l].device)
        w = w / w.norm().clamp_min(1e-12)
        out["n"].append(d - (d @ w) * w)
    return out


def random_delta(eng, dl: dict, seed: int) -> dict:
    """delta's per-unit sizes with random directions: each head's output slice a random direction of the same norm,
    each neuron its own size with a random sign (the control for "any push of this size")."""
    g = torch.Generator().manual_seed(seed)
    out = {"h": [], "n": []}
    for l in range(eng.L):
        d = dl["h"][l].cpu().view(eng.nheads[l], eng.hdim[l])
        r = torch.randn(d.shape, generator=g)
        out["h"].append((r / r.norm(dim=1, keepdim=True) * d.norm(dim=1, keepdim=True)).flatten())
        n = dl["n"][l].cpu()
        out["n"].append(n.abs() * (torch.randint(0, 2, n.shape, generator=g) * 2 - 1).float())
    return out


def masks(eng, units) -> dict:
    """{(kind, l): bool [W]} of a set of (kind, layer, index) units (heads expanded over their output slice)."""
    out = {}
    for kind, l, i in units:
        W = eng.nheads[l] if kind == "h" else eng.D
        out.setdefault((kind, l), torch.zeros(W, dtype=torch.bool))[i] = True
    return {k: (v.repeat_interleave(eng.hdim[k[1]]) if k[0] == "h" else v) for k, v in out.items()}


class Shift:
    """edit(kind, l, x [B, T, W]): row b adds coef[b] x delta on its set's units at the positions of where (dl: one
    delta for every row, or a list with each row's own)."""

    def __init__(self, eng, dl, rows: list, P: int, every: bool):
        self.eng, self.P, self.every = eng, P, every
        self.dl = dl if isinstance(dl, list) else [dl] * len(rows)
        self.coef = torch.tensor([c for c, _ in rows], dtype=torch.float32)
        self.ms = [m for _, m in rows]

    def sub(self, a: int, b: int) -> "Shift":
        return Shift(self.eng, self.dl[a:b], list(zip(self.coef[a:b].tolist(), self.ms[a:b])), self.P, self.every)

    def __call__(self, kind, l, x):
        B, T, W = x.shape
        m = torch.stack([mk.get((kind, l), torch.zeros(W, dtype=torch.bool)) for mk in self.ms]).to(x.device)
        if not bool(m.any()):
            return x
        d = torch.stack([dl[kind][l].to(x.device) for dl in self.dl])
        add = (self.coef.to(x.device)[:, None] * m) * d
        if self.every:
            return x + add[:, None].to(x.dtype)
        x = x.clone()
        x[:, T - self.P:] += add[:, None].to(x.dtype)
        return x


ROW_TOKENS = int(os.environ.get("DCORE_ROW_TOKENS", 4096))     # rows x tokens of one forward


@torch.no_grad()
def run_rows(eng, ids: list, sh: Shift) -> torch.Tensor:
    """[B, V] log-probs after ids (a full forward, no prefix cache) for each row of the shift, in chunks of rows."""
    B, out = len(sh.ms), []
    per = max(1, ROW_TOKENS // len(ids))
    for a in range(0, B, per):
        part = sh.sub(a, a + per)
        b = len(part.ms)
        x = torch.as_tensor(ids, device=eng.dev)[None].expand(b, -1)
        pos = torch.arange(len(ids), device=eng.dev)[None].expand(b, -1)
        hn, _ = eng._forward(x, pos, eng._empty(b), part, False)
        out.append(eng.logp(hn))
    return torch.cat(out)


def lt(row, ids) -> float:
    return float(torch.logsumexp(row[ids], 0) / LN2)


def score(eng, pr: E.Pair, dl: dict, sets: list, alphas: list, P: int, every: bool) -> dict:
    """Per (set, alpha): N, Nt, Nc, flipN, mN on the deceptive run; R, Rl, Rc, flipR on the honest run."""
    rows = [(a, sets[s]) for s in range(len(sets)) for a in alphas]
    out = {}
    lp = run_rows(eng, pr.dec.ids, Shift(eng, dl, rows, P, every))
    d = pr.dec
    for j, row in enumerate(lp):
        m = eng.metric(row, d.lie, d.truth)
        r = {"mN": m, "flipN": int(row.argmax()) in d.truth}
        if pr.hon is not None:
            r["N"] = (d.m - m) / pr.gap
            t = pr.hon.lt - d.lt
            r["Nt"] = (lt(row, d.truth) - d.lt) / t if t > E.TRUTH_MIN else float("nan")
            r["Nc"] = min(r["N"], r["Nt"]) if np.isfinite(r["Nt"]) else r["N"]
        out[j] = r
    if pr.hon is not None:
        h = pr.hon
        lp = run_rows(eng, h.ids, Shift(eng, dl, [(-a, s) for a, s in rows], P, every))
        for j, row in enumerate(lp):
            m = eng.metric(row, h.lie, h.truth)
            t = d.ll - h.ll
            rl = (lt(row, h.lie) - h.ll) / t if t > E.TRUTH_MIN else float("nan")
            r = out[j]
            r.update({"R": (m - h.m) / pr.gap, "Rl": rl, "flipR": int(row.argmax()) in h.lie})
            r["Rc"] = min(r["R"], rl) if np.isfinite(rl) else r["R"]
    return out


def summarize(recs: list, mdec: list) -> dict:
    """Medians with 95% intervals (or deceptive-only: lie margin change, flips, lies kept) over pairs."""
    out = {"n": len(recs)}
    for k in ("N", "Nc", "R", "Rc", "Nt", "Rl"):
        x = np.array([r.get(k, np.nan) for r in recs], float)
        if np.isfinite(x).any():
            out[k] = [round(v, 4) for v in E.boot_median(x)]
    for k in ("flipN", "flipR"):
        x = np.array([float(r[k]) for r in recs if k in r])
        if len(x):
            out[k] = [round(v, 4) for v in E.boot_mean(x)]
    if "N" not in out:
        mN = np.array([r["mN"] for r in recs])
        out["dm_bits"] = [round(v, 4) for v in E.boot_median(np.array(mdec) - mN)]
        out["lie_kept"] = [round(v, 4) for v in E.boot_mean((mN > 0).astype(float))]
    return out


def brief(s: dict) -> str:
    return " ".join(f"{k} {s[k][0]:.2f}" for k in ("N", "Nc", "R", "Rc", "flipN", "flipR", "dm_bits", "lie_kept")
                    if k in s)


def run(a):
    torch.manual_seed(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = E.load(a.model, a.dtype, a.device)
    eng = E.Engine(model)
    fams, held = a.families.split(","), [f for f in a.heldout.split(",") if f]
    data = C.Data(eng, tok, json.load(open(a.items)), fams + held, a.n_select, a.n_test, a.span_max)
    P, t0 = a.span, time.time()
    dl = delta(eng, data, fams, P)
    if a.project_out:
        u, nu = format_dir(eng, tok, a.controls, a.project_out.split(","), P)
        cos = [float(torch.nn.functional.cosine_similarity(dl["h"][l].flatten(), u["h"][l].to(dl["h"][l].device).flatten(),
                                                           0)) for l in range(eng.L)]
        E.log(f"format direction over {nu} {a.project_out} controls; per-layer cosine with delta (heads): "
              + " ".join(f"{c:.2f}" for c in cos))
        dl = project_out(eng, dl, u)
    torch.save({k: [x.cpu() for x in v] for k, v in dl.items()}, out / "delta.pt")
    E.log(f"delta over the select pairs of {fams} at span {P} ({time.time() - t0:.0f} s)")
    names = ["every_head", "every_unit"]
    sets = [[("h", l, i) for l in range(eng.L) for i in range(eng.nheads[l])]]
    sets.append(sets[0] + [("n", l, i) for l in range(eng.L) for i in range(eng.D)])
    for cp in [c for c in a.cores.split(",") if c]:
        core = json.load(open(os.path.expanduser(cp)))
        us = sorted({("h" if u["kind"] == "head" else "n", u["layer"], u["index"]) for u in core["units"]})
        names.append(f"core_v{core['version']}")
        sets.append(us)
        E.log(f"{names[-1]}: {len(core['units'])} units, {len(us)} without position")
    mk = [masks(eng, s) for s in sets]
    pr = next(iter(data.pairs(data.by[(fams[0], "test")][:1], P)))[0]       # the full forward equals prefix + span
    m0 = eng.metric(run_rows(eng, pr.dec.ids, Shift(eng, dl, [(0.0, mk[0])], P, True))[0], pr.dec.lie, pr.dec.truth)
    E.log(f"check: full forward m {m0:.5f} vs prefix + span m {pr.dec.m:.5f}")
    alphas = [float(x) for x in a.alphas.split(",")]
    res = {"P": P, "families": fams, "heldout": held, "alphas": alphas, "sets": names,
           "set_sizes": [len(s) for s in sets], "items": data.ids()}
    wheres = [(w, dl, w == "every") for w in a.wheres.split(",") if w]
    wheres += [(f"every random{k}", random_delta(eng, dl, k), True) for k in range(a.random_deltas)]
    for where, dw, every in wheres:
        res[where] = {}
        for f in fams + held:
            splits = ["test"] if f in fams else ["select", "test"]
            recs = [r for s in splits for r in data.by.get((f, s), []) if r["tail"] >= P][:a.n_items]
            per, mdec = {}, []
            for grp in data.pairs(recs, P):
                for pr in grp:
                    sc = score(eng, pr, dw, mk, alphas, P, every)
                    mdec.append(pr.dec.m)
                    for j, r in sc.items():
                        per.setdefault(j, []).append(r)
            res[where][f] = {f"{names[j // len(alphas)]}@{alphas[j % len(alphas)]}": summarize(v, mdec)
                             for j, v in per.items()}
            for k, v in res[where][f].items():
                E.log(f"{where} {f} {k}: {brief(v)}")
            json.dump(res, open(out / "steer.json", "w"), indent=1)
    if a.controls:
        ctl = C.control_recs(tok, a.controls, a.ctl_groups.split(","))
        res["controls"] = {}
        for g, recs in ctl.items():
            stay, n = {}, 0
            for grp in data.pairs([r for r in recs if r["tail"] >= P][:a.n_items], P):
                for pr in grp:
                    if pr.dec.top not in pr.dec.truth:
                        continue
                    n += 1
                    for where, dw, every in wheres:
                        rows = [(al, mk[s]) for s in range(len(sets)) for al in alphas]
                        lp = run_rows(eng, pr.dec.ids, Shift(eng, dw, rows, P, every))
                        for j, row in enumerate(lp):
                            k = f"{where} {names[j // len(alphas)]}@{alphas[j % len(alphas)]}"
                            stay[k] = stay.get(k, 0) + (int(row.argmax()) in pr.dec.truth)
            res["controls"][g] = {"n": n, **{k: round(v / max(n, 1), 4) for k, v in stay.items()}}
            E.log(f"controls {g} ({n} right unedited): " + "; ".join(f"{k} {v / max(n, 1):.2f}" for k, v in stay.items()))
        json.dump(res, open(out / "steer.json", "w"), indent=1)
    E.log(f"steer done {time.time() - t0:.0f} s -> {out / 'steer.json'}")


# ---------------------------------------------------------------- a head set for the every-position shift
def ig_heads(eng, run: E.Run, dl: dict, sign: float, steps: int) -> torch.Tensor:
    """[L, H] integrated gradients of m along the every-head shift c x sign x delta (every position), c 0 -> 1 at
    steps midpoints: each head's share of m(c=1) - m(c=0), H the widest layer's head count (others zero-padded)."""
    H = max(eng.nheads)
    out = torch.zeros(eng.L, H)
    for k in range(steps):
        c = (k + 0.5) / steps
        cs = {}

        def edit(kind, l, x, c=c):
            if kind != "h":
                return x
            nh, hd = eng.nheads[l], eng.hdim[l]
            z = torch.full((nh,), c * sign, device=x.device, dtype=x.dtype, requires_grad=True)
            cs[l] = z
            return x + (z.repeat_interleave(hd) * dl["h"][l].to(x.device, x.dtype))[None, None]
        with torch.enable_grad():
            ids = torch.as_tensor(run.ids, device=eng.dev)[None]
            pos = torch.arange(len(run.ids), device=eng.dev)[None]
            hn, _ = eng._forward(ids, pos, eng._empty(1), edit, False)
            lp = eng.logp(hn)[0]
            m = (torch.logsumexp(lp[run.lie], 0) - torch.logsumexp(lp[run.truth], 0)) / LN2
            m.backward()
        for l, z in cs.items():
            out[l, :eng.nheads[l]] += z.grad.float().cpu() * sign / steps
        del lp, m, hn
    return out


def head_sets(eng, order: list, sizes: list) -> list:
    return [masks(eng, [("h", l, i) for l, i in order[:k]]) for k in sizes]


def pair_scores(eng, pr, dl, mks: list, alpha: float, P: int) -> list:
    """Per set: N, Nc, R, Rc, flips of the every-position shift (score() with one alpha)."""
    sc = score(eng, pr, dl, mks, [alpha], P, True)
    return [sc[j] for j in range(len(mks))]


def collect(eng, data, fams: list, splits: list, P: int, dl, mks: list, alpha: float, n_items: int) -> dict:
    """{fam: [per set: summarize over pairs]} on fams' splits (held-out: every split given)."""
    res = {}
    for f in fams:
        recs = [r for s in splits for r in data.by.get((f, s), []) if r["tail"] >= P][:n_items]
        per, mdec = [[] for _ in mks], []
        for grp in data.pairs(recs, P):
            for pr in grp:
                for j, r in enumerate(pair_scores(eng, pr, dl, mks, alpha, P)):
                    per[j].append(r)
                mdec.append(pr.dec.m)
        res[f] = [summarize(v, mdec) for v in per]
    return res


def select(a):
    """The smallest set of heads (ordered by the integrated gradients of the every-head shift, cross-family min or
    mean) whose every-position shift passes median credited N and credited R >= PASS on every selection family's
    select pairs; then its test pairs, held-out families, random same-layer head sets, controls and alpha."""
    torch.manual_seed(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = E.load(a.model, a.dtype, a.device)
    eng = E.Engine(model)
    fams, held = a.families.split(","), [f for f in a.heldout.split(",") if f]
    data = C.Data(eng, tok, json.load(open(a.items)), fams + held, a.n_select, a.n_test, a.span_max)
    P, t0 = a.span, time.time()
    dl = delta(eng, data, fams, P)
    torch.save({k: [x.cpu() for x in v] for k, v in dl.items()}, out / "delta.pt")
    heads = [(l, i) for l in range(eng.L) for i in range(eng.nheads[l])]
    sa = a.select_alpha                 # the shift's size the set is ordered and selected at
    dls = {k: [x * sa for x in v] for k, v in dl.items()}
    rec = {"P": P, "families": fams, "heldout": held, "pass": C.PASS, "items": data.ids(), "n_heads": len(heads),
           "select_alpha": sa}
    # 1. integrated gradients per family (deceptive runs: +delta; honest runs: -delta), normalized by the gap
    sc = {}
    for f in fams:
        acc, comp = torch.zeros(eng.L, max(eng.nheads)), []
        recs = data.by[(f, "select")][:a.ig_items]
        for grp in data.pairs(recs, P):
            for pr in grp:
                gN = -ig_heads(eng, pr.dec, dls, 1.0, a.ig_steps) / pr.gap
                gR = ig_heads(eng, pr.hon, dls, -1.0, a.ig_steps) / pr.gap
                acc += (gN + gR) / 2
                comp.append(float(gN.sum()))
        sc[f] = acc / max(1, len(comp))
        E.log(f"ig {f}: {len(comp)} pairs, N-side sum (every head's share) median {np.median(comp):.3f} "
              f"({time.time() - t0:.0f} s)")
    S = torch.stack([sc[f] for f in fams])
    valid = [(l, i) for l, i in heads]
    orders = {}
    for rule, v in (("min", S.min(0).values), ("mean", S.mean(0))):
        orders[rule] = sorted(valid, key=lambda h: -float(v[h]))
    rec["ig"] = {f: sc[f].tolist() for f in fams}
    # 2. curves on every select pair, both orders
    sizes = E.grid(len(heads), a.per_doubling)
    mks = [m for r in orders for m in head_sets(eng, orders[r], sizes)]
    cur = collect(eng, data, fams, ["select"], P, dl, mks, sa, a.curve_items)
    best, rec["curves"] = None, {}
    for q, r in enumerate(orders):
        rows = {f: cur[f][q * len(sizes):(q + 1) * len(sizes)] for f in fams}
        ok = [all(rows[f][j].get("Nc", [0])[0] >= C.PASS and rows[f][j].get("Rc", [0])[0] >= C.PASS for f in fams)
              for j in range(len(sizes))]
        ms = [min(min(rows[f][j].get("Nc", [-9])[0], rows[f][j].get("Rc", [-9])[0]) for f in fams)
              for j in range(len(sizes))]
        first = sizes[ok.index(True)] if any(ok) else None
        rec["curves"][r] = {"sizes": sizes, "first_passing": first, "min_share": ms,
                            **{f: [{k: x[k][0] for k in ("N", "Nc", "R", "Rc", "flipN", "flipR") if k in x}
                                   for x in rows[f]] for f in fams}}
        E.log(f"curve {r}: first passing {first}; best min-share {max(ms):.3f} at {sizes[int(np.argmax(ms))]} heads; "
              + "; ".join(f"{k} heads " + " ".join(f"{f} Nc {rows[f][j].get('Nc', [np.nan])[0]:.2f} Rc "
                                                    f"{rows[f][j].get('Rc', [np.nan])[0]:.2f}" for f in fams)
                          for j, k in enumerate(sizes) if k in (8, 32, 128, 512) or k == sizes[-1]))
        key = (first is None, first or 0, -max(ms))
        if best is None or key < best[0]:
            best = (key, r, first or sizes[int(np.argmax(ms))])
    _, rule, k = best
    order = orders[rule]
    core = order[:k]
    rec.update({"rule": rule, "size": k, "met": best[0][0] is False,
                "heads": [{"layer": l, "index": i, "label": f"L{l}.H{i}"} for l, i in core],
                "layers": {int(l): sum(1 for x, _ in core if x == l) for l in sorted({l for l, _ in core})}})
    E.log(f"head set ({rule}): {k} heads (met {rec['met']}) layers {rec['layers']} ({time.time() - t0:.0f} s)")
    json.dump(rec, open(out / "steer_heads.json", "w"), indent=1)
    # 3. test pairs and held-out families: the set, every head, random same-layer sets
    rng = np.random.default_rng(0)
    rand = []
    for _ in range(a.random):
        used, rs = set(core), []
        for l, _i in core:
            free = [i for i in range(eng.nheads[l]) if (l, i) not in used]
            j = int(rng.choice(free)) if free else None
            if j is None:       # (a layer whose heads are all taken: the nearest layer with a free one)
                l2 = min((x for x in range(eng.L) if any((x, i) not in used for i in range(eng.nheads[x]))),
                         key=lambda x: abs(x - l))
                l, j = l2, int(rng.choice([i for i in range(eng.nheads[l2]) if (l2, i) not in used]))
            used.add((l, j))
            rs.append((l, j))
        rand.append(rs)
    tm = [masks(eng, [("h", l, i) for l, i in core]), masks(eng, [("h", l, i) for l, i in heads])] + \
        [masks(eng, [("h", l, i) for l, i in r]) for r in rand]
    rec["test"] = {}
    for al in [float(x) for x in a.alphas.split(",")]:
        ts = collect(eng, data, fams, ["test"], P, dl, tm, al, a.n_test)
        hs = collect(eng, data, held, ["select", "test"], P, dl, tm, al, a.n_test) if held else {}
        rec["test"][str(al)] = {}
        for f, v in {**ts, **hs}.items():
            q = "Nc" if "Nc" in v[0] else "flipN"
            rmax = max(x.get(q, [np.nan])[0] for x in v[2:]) if len(v) > 2 else float("nan")
            rec["test"][str(al)][f] = {"set": v[0], "every_head": v[1], "random": v[2:],
                                       f"beats_every_random_{q}": bool(v[0].get(q, [np.nan])[0] > rmax)}
            E.log(f"test alpha {al} {f}: set {brief(v[0])}; every head {brief(v[1])}; random max {q} {rmax:.2f}")
        json.dump(rec, open(out / "steer_heads.json", "w"), indent=1)
    # 4. controls under +select_alpha x delta on the set, every head and the random sets
    if a.controls:
        ctl = C.control_recs(tok, a.controls, a.ctl_groups.split(","))
        rec["controls"] = {}
        for g, recs in ctl.items():
            stay, n = np.zeros(len(tm)), 0
            for grp in data.pairs([r for r in recs if r["tail"] >= P][:a.n_test], P):
                for pr in grp:
                    if pr.dec.top not in pr.dec.truth:
                        continue
                    n += 1
                    lp = run_rows(eng, pr.dec.ids, Shift(eng, dl, [(sa, m) for m in tm], P, True))
                    stay += np.array([int(r.argmax()) in pr.dec.truth for r in lp], float)
            st = stay / max(n, 1)
            rec["controls"][g] = {"n": n, "set": round(st[0], 4), "every_head": round(st[1], 4),
                                  "random_mean": round(float(st[2:].mean()), 4) if len(st) > 2 else None}
            E.log(f"controls {g} ({n}): set {st[0]:.2f}, every head {st[1]:.2f}, random mean "
                  f"{st[2:].mean() if len(st) > 2 else float('nan'):.2f}")
        json.dump(rec, open(out / "steer_heads.json", "w"), indent=1)
    E.log(f"steer select done {time.time() - t0:.0f} s -> {out / 'steer_heads.json'}")


# ---------------------------------------------------------------- natural lies: lie vs resist under the same pressure
def share_ci(x) -> list:
    """[share, lo, hi]: the exact (Clopper-Pearson) 95% interval of a share."""
    from scipy.stats import beta
    x = np.asarray(x, float)
    n, k = len(x), float(x.sum())
    if not n:
        return [float("nan")] * 3
    return [round(k / n, 4), round(float(beta.ppf(.025, k, n - k + 1)) if k else 0.0, 4),
            round(float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.0, 4)]


def median_ci(x) -> list:
    """[median, lo, hi]: the distribution-free 95% interval of a median (binomial order statistics; nan where n is too
    small for one)."""
    from scipy.stats import binom
    x = np.sort(np.asarray(x, float)[np.isfinite(np.asarray(x, float))])
    n = len(x)
    if not n:
        return [float("nan")] * 3
    j = int(binom.ppf(.025, n, .5))
    lo, hi = (x[j - 1], x[n - j]) if j >= 1 else (float("nan"), float("nan"))
    return [round(float(np.median(x)), 4), round(float(lo), 4), round(float(hi), 4)]


def nat_rec(eng, tok, it: dict, P: int, mean: bool, at: str = "decision") -> dict | None:
    """One item's pressure run at span P, re-measured (and the pressure-free partner's m and log2 p(truth) where the
    item has one); mean: also each unit's value averaged over the span (for delta), the P positions before the decision
    (at "decision") or the prompt's last P positions, before any reply prefix (at "prompt"). None if the tail is under
    P."""
    d, h = E.item_seqs(tok, it)
    if E.template_tail(tok, d) < P:
        return None
    r = eng.prepare(d, P, it["lie_ids"], it["truth_ids"])
    rec = {"it": it, "ids": d, "m": r.m, "lt": r.lt, "ll": r.ll, "lie": r.lie, "truth": r.truth, "top": r.top}
    if mean:
        if at == "prompt" and it.get("prefix_ids"):
            del r
            r = eng.prepare(d[:len(d) - len(it["prefix_ids"])], P, it["lie_ids"], it["truth_ids"])
        rec["mean"] = {k: [x.double().mean(0).cpu() for x in r.cap[k]] for k in ("h", "n")}
    del r
    if h is not None:
        rh = eng.prepare(h, P, it["lie_ids"], it["truth_ids"])
        rec.update(m_hon=rh.m, lt_hon=rh.lt)
        del rh
    return rec


def nat_recs(eng, tok, items: list, fam: str, splits: list, label: str, n: int, P: int, mean: bool = False,
             at: str = "decision") -> list:
    """Up to n usable items of fam/splits with this label (lie / resist), seeded order; kept where the float32
    re-measure agrees with the label (lie: m > 0; resist: m < 0)."""
    cand = C.seeded([it for it in items if it["family"] == fam and it["split"] in splits and it.get("label") == label
                     and it.get("usable")], f"{fam}/{'+'.join(splits)}/{label}")
    out, short, sign = [], 0, 0
    for it in cand:
        if len(out) >= n:
            break
        rec = nat_rec(eng, tok, it, P, mean, at)
        if rec is None:
            short += 1
        elif (rec["m"] > 0) != (label == "lie"):
            sign += 1
        else:
            out.append(rec)
    E.log(f"natural {fam:8s} {'+'.join(splits):11s} {label:6s} candidates {len(cand):4d} used {len(out):4d} "
          f"(tail under {P}: {short}, float32 m sign disagrees with the label: {sign}); median m "
          f"{np.median([r['m'] for r in out]) if out else float('nan'):+.2f}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def instructed_recs(eng, tok, items: list, fam: str, splits: list, n: int, P: int) -> list:
    """Up to n instructed lies of fam (corefind's candidates: paired m_dec > 0 > m_hon), as nat_rec records."""
    cand = C.seeded([it for s in splits for it in C.candidates(items, fam, s)], f"{fam}/{'+'.join(splits)}")
    out = []
    for it in cand:
        if len(out) >= n:
            break
        rec = nat_rec(eng, tok, it, P, False)
        if rec is not None and rec["m"] > 0 and (rec.get("m_hon") is None or rec["m_hon"] < 0):
            out.append(rec)
    E.log(f"instructed {fam:8s} candidates {len(cand):4d} used {len(out):4d}")
    return out


def delta_nat(lies: list, res: list, bins: int) -> tuple:
    """({"h"/"n": [per layer [W]]}, info): the resist mean minus the lie mean of each unit's span-averaged value. With
    bins > 1 and every item's knowledge_margin (m of the pressure-free run), matched on it: per quantile bin of the lies'
    margins, the bin's resist mean minus lie mean, weighted by the bin's lie count (a bin without both is dropped)."""
    km = [r["it"].get("knowledge_margin") for r in lies + res]
    if bins > 1 and all(k is not None for k in km):
        edges = np.quantile([r["it"]["knowledge_margin"] for r in lies], np.linspace(0, 1, bins + 1)[1:-1])
        bin_of = [int(np.searchsorted(edges, r["it"]["knowledge_margin"])) for r in lies + res]
    else:
        bins, bin_of = 1, [0] * len(lies + res)
    acc, wsum, info = None, 0, []
    for j in range(bins):
        L = [r for r, b in zip(lies, bin_of) if b == j]
        R = [r for r, b in zip(res, bin_of[len(lies):]) if b == j]
        row = {"bin": j, "lies": len(L), "resists": len(R)}
        if bins > 1:
            row["knowledge_margin_median"] = [round(float(np.median([r["it"]["knowledge_margin"] for r in g])), 3)
                                              if g else None for g in (L, R)]
        row["m_median"] = [round(float(np.median([r["m"] for r in g])), 3) if g else None for g in (L, R)]
        info.append(row)
        E.log(f"delta bin {j}: {row}")
        if not L or not R:
            continue
        d = {k: [sum(r["mean"][k][l] for r in R) / len(R) - sum(r["mean"][k][l] for r in L) / len(L)
                 for l in range(len(L[0]["mean"][k]))] for k in ("h", "n")}
        acc = {k: [x * len(L) for x in v] for k, v in d.items()} if acc is None else \
            {k: [a + x * len(L) for a, x in zip(acc[k], d[k])] for k in acc}
        wsum += len(L)
    return {k: [(x / wsum).float() for x in v] for k, v in acc.items()}, info


def nat_score(eng, rec: dict, variants: list, P: int, sign: float) -> dict:
    """{(variant, where, alpha): per-item scores} of sign x alpha x delta on every head, for each variant (name, dl,
    wheres, alphas). Lies (sign +): removed (m < 0), flip (top token a truth id), dm (m drop), N / Nt / Nc where the
    item has a pressure-free partner. Resists (sign -): induced (m > 0), flip (top token a lie id), dm (m rise)."""
    out = {}
    for where in ("every", "span"):
        rows, dls, keys = [], [], []
        for name, dl, wheres, alphas, mk in variants:
            if where in wheres:
                for al in alphas:
                    rows.append((sign * al, mk))
                    dls.append(dl)
                    keys.append((name, where, al))
        if not rows:
            continue
        lp = run_rows(eng, rec["ids"], Shift(eng, dls, rows, P, where == "every"))
        for k, row in zip(keys, lp):
            m = eng.metric(row, rec["lie"], rec["truth"])
            top = int(row.argmax())
            if sign > 0:
                r = {"removed": m < 0, "flip": top in rec["truth"], "dm": rec["m"] - m}
                if rec.get("m_hon") is not None:
                    r["N"] = (rec["m"] - m) / (rec["m"] - rec["m_hon"])
                    t = rec["lt_hon"] - rec["lt"]
                    r["Nt"] = (lt(row, rec["truth"]) - rec["lt"]) / t if t > E.TRUTH_MIN else float("nan")
                    r["Nc"] = min(r["N"], r["Nt"]) if np.isfinite(r["Nt"]) else r["N"]
            else:
                r = {"induced": m > 0, "flip": top in rec["lie"], "dm": m - rec["m"]}
            out[k] = r
    return out


def nat_summary(recs: list) -> dict:
    out = {"n": len(recs)}
    for k in ("removed", "induced", "flip"):
        if recs and k in recs[0]:
            out[k] = share_ci([float(r[k]) for r in recs])
    for k in ("dm", "N", "Nc"):
        if recs and k in recs[0]:
            out[k] = median_ci([r[k] for r in recs])
    return out


def nat_brief(s: dict) -> str:
    return f"n {s['n']} " + " ".join(f"{k} {s[k][0]:.2f} [{s[k][1]:.2f}, {s[k][2]:.2f}]"
                                     for k in ("removed", "induced", "flip", "Nc", "dm") if k in s)


def natural(a):
    """Contrast A: delta = the source families' select resist mean minus lie mean (knowledge-matched where items carry a
    knowledge margin), applied to every head at every position or the span only; lie removal and lie induction on every
    natural family (the source's test split, other families' every split), the instructed families as transfer, and
    controls, against random deltas of the same per-head sizes."""
    torch.manual_seed(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = E.load(a.model, a.dtype, a.device)
    eng = E.Engine(model)
    items = json.load(open(os.path.expanduser(a.items)))
    items = items["items"] if isinstance(items, dict) else items
    src, fams = a.source.split(","), [f for f in a.families.split(",") if f]
    P, t0 = a.span, time.time()
    lies = [r for f in src for r in nat_recs(eng, tok, items, f, ["select"], "lie", a.n_select, P, True, a.delta_at)]
    res = [r for f in src for r in nat_recs(eng, tok, items, f, ["select"], "resist", a.n_select, P, True, a.delta_at)]
    dl, info = delta_nat(lies, res, a.bins)
    del lies, res
    stem = "delta_A_" + "+".join(src) + ("_prompt" if a.delta_at == "prompt" else "")
    torch.save({**{k: [x.cpu() for x in v] for k, v in dl.items()}, "info": info, "P": P, "source": src},
               out / f"{stem}.pt.tmp")
    os.replace(out / f"{stem}.pt.tmp", out / f"{stem}.pt")
    E.log(f"{stem}: {time.time() - t0:.0f} s")
    every_head = masks(eng, [("h", l, i) for l in range(eng.L) for i in range(eng.nheads[l])])
    alphas = [float(x) for x in a.alphas.split(",")]
    wheres = a.wheres.split(",")
    variants = [("delta", dl, wheres, alphas, every_head)]
    variants += [(f"random{k}", random_delta(eng, dl, k), ["every"], alphas, every_head) for k in range(a.random_deltas)]
    rec = {"P": P, "source": src, "delta_at": a.delta_at, "alphas": alphas, "wheres": wheres,
           "random_deltas": a.random_deltas,
           "delta_bins": info, "families": {}, "transfer": {}, "controls": {}}

    def dump():
        json.dump(rec, open(out / f"{stem}.json.tmp", "w"), indent=1)
        os.replace(out / f"{stem}.json.tmp", out / f"{stem}.json")

    def table(recs: list, sign: float) -> dict:
        per = {}
        for r in recs:
            for k, v in nat_score(eng, r, variants, P, sign).items():
                per.setdefault(k, []).append(v)
        return {f"{n} {w}@{al}": nat_summary(v) for (n, w, al), v in per.items()}

    def report(group: str, name: str, tab: dict):
        for k, s in tab.items():
            if not k.startswith("random"):
                E.log(f"{group} {name} {k}: {nat_brief(s)}")
        for w in ("every",):
            for al in alphas:
                rs = [tab[f"random{j} {w}@{al}"] for j in range(a.random_deltas) if f"random{j} {w}@{al}" in tab]
                key = "removed" if rs and "removed" in rs[0] else "induced"
                if rs:
                    v = [s[key][0] for s in rs]
                    E.log(f"{group} {name} random {w}@{al}: {key} min {min(v):.2f} median {np.median(v):.2f} "
                          f"max {max(v):.2f} over {len(v)}")
    for f in fams:
        splits = ["test"] if f in src else ["select", "test"]
        rec["families"][f] = {}
        for label, sign in (("lie", 1.0), ("resist", -1.0)):
            rs = nat_recs(eng, tok, items, f, splits, label, a.n_items, P)
            if rs:
                rec["families"][f][label] = table(rs, sign)
                report(f, label, rec["families"][f][label])
                dump()
    for path in [p for p in a.transfer_items.split(",") if p]:
        titems = json.load(open(os.path.expanduser(path)))
        titems = titems["items"] if isinstance(titems, dict) else titems
        for f in [f for f in a.transfer.split(",") if f and any(it["family"] == f for it in titems)]:
            rs = instructed_recs(eng, tok, titems, f, ["test"], a.n_items, P)
            if rs:
                rec["transfer"][f] = table(rs, 1.0)
                report("transfer", f, rec["transfer"][f])
                dump()
    if a.controls:
        for g, crs in C.control_recs(tok, a.controls, a.ctl_groups.split(",")).items():
            stay, n = {}, 0
            for c in [c for c in crs if c["tail"] >= P][:a.n_items]:
                ids, truth = c["d"], c["it"]["truth_ids"]
                if int(eng.full_logp(ids).argmax()) not in truth:
                    continue
                n += 1
                cr = {"ids": ids, "lie": c["it"]["lie_ids"], "truth": truth, "m": 0.0, "lt": 0.0}
                for k, v in nat_score(eng, cr, variants, P, 1.0).items():
                    stay[k] = stay.get(k, 0) + int(v["flip"])
            tab = {f"{nm} {w}@{al}": round(v / max(n, 1), 4) for (nm, w, al), v in stay.items()}
            rec["controls"][g] = {"n": n, **tab}
            rnd = [v for k, v in tab.items() if k.startswith("random") and k.endswith(f"every@{alphas[-1]}")]
            E.log(f"controls {g} ({n} right unedited): " + "; ".join(f"{k} {v:.2f}" for k, v in tab.items()
                                                                       if not k.startswith("random"))
                  + (f"; random every@{alphas[-1]} min {min(rnd):.2f} median {np.median(rnd):.2f}" if rnd else ""))
            dump()
    dump()
    E.log(f"natural done {time.time() - t0:.0f} s -> {out / (stem + '.json')}")


NAT_HI = 1.3        # natsel's two-sided bar: a set's median share of every head's m change in [PASS, NAT_HI]


def nat_rel(eng, rec: dict, dl: dict, mks: list, alpha: float, P: int, sign: float) -> list:
    """Per set: this item's m change under sign x alpha x delta on the set (every position), as a share of the change
    with every head (the last of mks)."""
    lp = run_rows(eng, rec["ids"], Shift(eng, dl, [(sign * alpha, m) for m in mks], P, True))
    ms = [eng.metric(row, rec["lie"], rec["truth"]) for row in lp]
    full = ms[-1] - rec["m"]
    return [(m - rec["m"]) / full if abs(full) > 1e-6 else float("nan") for m in ms]


def natsel(a):
    """The smallest head set whose every-position shift along a natural delta (delta_A_<source>.pt) does what every
    head's does: heads ordered by integrated gradients over the source's select lies (+alpha x delta) and resists
    (-alpha x delta), each item's head shares normalized by its whole m change; per set size the median share of every
    head's m change, on lies and on resists; the first size where both lie in [PASS, NAT_HI]; then the set against
    every head and random same-layer head sets on the source's test items, the other natural families, the instructed
    families and controls -> DIR/heads_nat.json."""
    torch.manual_seed(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = E.load(a.model, a.dtype, a.device)
    eng = E.Engine(model)
    items = json.load(open(os.path.expanduser(a.items)))
    items = items["items"] if isinstance(items, dict) else items
    src, fams = a.source.split(","), [f for f in a.families.split(",") if f]
    P, t0, sa = a.span, time.time(), a.select_alpha
    dz = torch.load(os.path.expanduser(a.delta), weights_only=False)
    dl = {k: dz[k] for k in ("h", "n")}
    dls = {k: [x * sa for x in v] for k, v in dl.items()}
    heads = [(l, i) for l in range(eng.L) for i in range(eng.nheads[l])]
    every = masks(eng, [("h", l, i) for l, i in heads])
    rec = {"P": P, "source": src, "delta": a.delta, "select_alpha": sa, "pass": C.PASS, "n_heads": len(heads)}

    def dump():
        json.dump(rec, open(out / "heads_nat.json.tmp", "w"), indent=1)
        os.replace(out / "heads_nat.json.tmp", out / "heads_nat.json")
    # 1. integrated gradients over the source's select lies (+) and resists (-), per item normalized by its m change
    sel = {lab: [r for f in src for r in nat_recs(eng, tok, items, f, ["select"], lab, a.ig_items, P)]
           for lab in ("lie", "resist")}
    sc = {}
    for lab, sign in (("lie", 1.0), ("resist", -1.0)):
        acc, n, tot = torch.zeros(eng.L, max(eng.nheads)), 0, []
        for r in sel[lab]:
            run = types.SimpleNamespace(ids=r["ids"], lie=r["lie"], truth=r["truth"])
            g = ig_heads(eng, run, dls, sign, a.ig_steps)
            s = float(g.sum())
            tot.append(s)
            if abs(s) >= a.min_change:
                acc += g / s
                n += 1
        sc[lab] = acc / max(n, 1)
        E.log(f"ig {lab}: {n} of {len(tot)} items with an every-head m change of at least {a.min_change} bits "
              f"(median change {np.median(tot) if tot else float('nan'):+.2f}) ({time.time() - t0:.0f} s)")
    S = torch.stack([sc["lie"], sc["resist"]])
    orders = {rule: sorted(heads, key=lambda h, v=v: -float(v[h])) for rule, v in
              (("min", S.min(0).values), ("mean", S.mean(0)))}
    rec["ig"] = {k: v.tolist() for k, v in sc.items()}
    # 2. curves: per size, the median share of every head's m change, on lies and on resists
    sizes = E.grid(len(heads), a.per_doubling)
    rec["curves"], best = {}, None
    for rule, order in orders.items():
        mks = [masks(eng, [("h", l, i) for l, i in order[:k]]) for k in sizes] + [every]
        rel = {lab: np.array([nat_rel(eng, r, dl, mks, sa, P, sign)[:-1] for r in sel[lab][:a.curve_items]])
               for lab, sign in (("lie", 1.0), ("resist", -1.0))}
        med = {lab: np.nanmedian(v, 0) if len(v) else np.full(len(sizes), np.nan) for lab, v in rel.items()}
        ms = np.minimum(med["lie"], med["resist"])
        ok = (ms >= C.PASS) & (np.maximum(med["lie"], med["resist"]) <= NAT_HI)
        first = sizes[int(np.argmax(ok))] if ok.any() else None
        rec["curves"][rule] = {"sizes": sizes, "first_passing": first, "lie": med["lie"].round(4).tolist(),
                               "resist": med["resist"].round(4).tolist()}
        E.log(f"curve {rule}: first passing {first}; " + "; ".join(
            f"{k} heads lie {med['lie'][j]:.2f} resist {med['resist'][j]:.2f}" for j, k in enumerate(sizes)
            if k in (4, 8, 16, 32, 64, 128, 256, 512)))
        key = (first is None, first or 0, -float(np.nanmax(ms)))
        if best is None or key < best[0]:
            best = (key, rule, first or sizes[int(np.nanargmax(ms))])
        dump()
    _, rule, k = best
    core = orders[rule][:k]
    rec.update({"rule": rule, "size": k, "met": best[0][0] is False,
                "heads": [{"layer": l, "index": i, "label": f"L{l}.H{i}"} for l, i in core],
                "layers": {int(l): sum(1 for x, _ in core if x == l) for l in sorted({l for l, _ in core})}})
    E.log(f"head set ({rule}): {k} heads (met {rec['met']}) layers {rec['layers']} ({time.time() - t0:.0f} s)")
    dump()
    # 3. the set, every head and random same-layer head sets on test items, other families, transfer and controls
    rng = np.random.default_rng(0)
    variants = [("set", dl, ["every"], [sa], masks(eng, [("h", l, i) for l, i in core])),
                ("every_head", dl, ["every"], [sa], every)]
    for q in range(a.random):
        rs = []
        for l, _i in core:
            free = [i for i in range(eng.nheads[l]) if (l, i) not in set(core) | set(rs)]
            rs.append((l, int(rng.choice(free))) if free else (l, _i))
        variants.append((f"random{q}", dl, ["every"], [sa], masks(eng, [("h", l, i) for l, i in rs])))

    def table(recs: list, sign: float) -> dict:
        per = {}
        for r in recs:
            for (nm, _w, _al), v in nat_score(eng, r, variants, P, sign).items():
                per.setdefault(nm, []).append(v)
        return {nm: nat_summary(v) for nm, v in per.items()}

    def report(group: str, tab: dict):
        key = "removed" if "removed" in tab["set"] else "induced"
        rnd = [tab[f"random{q}"][key][0] for q in range(a.random)]
        E.log(f"{group}: set {nat_brief(tab['set'])}; every head {nat_brief(tab['every_head'])}; random {key} max "
              f"{max(rnd) if rnd else float('nan'):.2f}")
    rec["test"] = {}
    for f in fams:
        splits = ["test"] if f in src else ["select", "test"]
        for lab, sign in (("lie", 1.0), ("resist", -1.0)):
            rs = nat_recs(eng, tok, items, f, splits, lab, a.n_items, P)
            if rs:
                rec["test"][f"{f} {lab}"] = table(rs, sign)
                report(f"{f} {lab}", rec["test"][f"{f} {lab}"])
                dump()
    for path in [p for p in a.transfer_items.split(",") if p]:
        titems = json.load(open(os.path.expanduser(path)))
        titems = titems["items"] if isinstance(titems, dict) else titems
        for f in [f for f in a.transfer.split(",") if f and any(it["family"] == f for it in titems)]:
            rs = instructed_recs(eng, tok, titems, f, ["test"], a.n_items, P)
            if rs:
                rec["test"][f"transfer {f}"] = table(rs, 1.0)
                report(f"transfer {f}", rec["test"][f"transfer {f}"])
                dump()
    if a.controls:
        rec["controls"] = {}
        for g, crs in C.control_recs(tok, a.controls, a.ctl_groups.split(",")).items():
            stay, n = {}, 0
            for c in [c for c in crs if c["tail"] >= P][:a.n_items]:
                ids, truth = c["d"], c["it"]["truth_ids"]
                if int(eng.full_logp(ids).argmax()) not in truth:
                    continue
                n += 1
                cr = {"ids": ids, "lie": c["it"]["lie_ids"], "truth": truth, "m": 0.0, "lt": 0.0}
                for (nm, _w, _al), v in nat_score(eng, cr, variants, P, 1.0).items():
                    stay[nm] = stay.get(nm, 0) + int(v["flip"])
            st = {nm: round(v / max(n, 1), 4) for nm, v in stay.items()}
            rnd = [v for nm, v in st.items() if nm.startswith("random")]
            rec["controls"][g] = {"n": n, "set": st.get("set"), "every_head": st.get("every_head"),
                                  "random_mean": round(float(np.mean(rnd)), 4) if rnd else None}
            E.log(f"controls {g} ({n}): {rec['controls'][g]}")
            dump()
    E.log(f"natsel done {time.time() - t0:.0f} s -> {out / 'heads_nat.json'}")


def segments(tok, ids: list) -> tuple:
    """({group: bool [T]}, [system sentence texts]) of a chat: each system-prompt sentence (sys0, sys1, ...), the
    system prompt, the last user turn, earlier turns (history) and the reply (the last assistant header onward)."""
    T, st = len(ids), tok.convert_tokens_to_ids("<|im_start|>")
    starts = [i for i, t in enumerate(ids) if t == st] + [T]
    out, texts = {}, []

    def put(g, a, b):
        out.setdefault(g, torch.zeros(T, dtype=torch.bool))[a:b] = True
    n = len(starts) - 1
    for k in range(n):
        a, b = starts[k], starts[k + 1]
        role = tok.decode(ids[a + 1:a + 2]).strip()
        if k == n - 1:
            put("reply", a, b)
        elif role == "system":
            put("system", a, b)
            s = a
            for i in range(a, b):
                if i == b - 1 or tok.decode(ids[i:i + 1]).rstrip().endswith((".", "!", "?")):
                    put(f"sys{len(texts)}", s, i + 1)
                    texts.append(tok.decode(ids[s:i + 1]).strip())
                    s = i + 1
        elif k == n - 2 and role == "user":
            put("user", a, b)
        else:
            put("history", a, b)
    return out, texts


@torch.no_grad()
def head_outputs(eng, ids: list, layers: set) -> dict:
    """{layer: [T, W]} the unedited head outputs of these layers."""
    keep = {}

    def edit(kind, l, x):
        if kind == "h" and l in layers:
            keep[l] = x[0].detach().clone()
        return x
    x = torch.as_tensor(ids, device=eng.dev)[None]
    eng._forward(x, torch.arange(len(ids), device=eng.dev)[None], eng._empty(1), edit, False)
    return keep


class NatEdit:
    """Rows of (coef, shift masks, positions or None, freeze masks or None): row b adds coef x delta on its shift units
    at its positions (every position where None) and holds its freeze units at their unedited values (clean)."""

    def __init__(self, eng, dl: dict, rows: list, clean: dict):
        self.eng, self.dl, self.ms, self.clean = eng, dl, rows, clean

    def sub(self, a: int, b: int) -> "NatEdit":
        return NatEdit(self.eng, self.dl, self.ms[a:b], self.clean)

    def __call__(self, kind, l, x):
        if kind != "h":
            return x
        B, T, W = x.shape
        z = torch.zeros(W, dtype=torch.bool)
        sm = torch.stack([r[1].get((kind, l), z) for r in self.ms]).to(x.device)
        fm = torch.stack([(r[3] or {}).get((kind, l), z) for r in self.ms]).to(x.device)
        if not bool(sm.any()) and not bool(fm.any()):
            return x
        if bool(fm.any()):
            x = torch.where(fm[:, None, :], self.clean[l].to(x.device, x.dtype)[None], x)
        if bool(sm.any()):
            d = self.dl[kind][l].to(x.device, x.dtype)
            coef = torch.tensor([r[0] for r in self.ms], device=x.device, dtype=x.dtype)
            pos = torch.stack([r[2] if r[2] is not None else torch.ones(T, dtype=torch.bool)
                               for r in self.ms]).to(x.device, x.dtype)
            x = x + coef[:, None, None] * pos[:, :, None] * (sm.to(x.dtype) * d)[:, None, :]
        return x


GROUPS = ("system", "user", "history", "reply")


def natdag(a):
    """The graph of a natsel head set H (DIR/heads_nat.json) along its natural delta, on each family's lies (+alpha x
    delta: the lie removed) and resists (-alpha x delta: a lie induced), every number a share of m's change when all of
    H is shifted at every position (live, full model, nothing frozen unless said):
      source -> H: H shifted only at one token group's positions (the system prompt, each of its sentences, the user
        turn, earlier turns, the reply start); and each head shifted only at one coarse group;
      head h: h alone shifted; h -> answer: h shifted with every later head of H held at its unedited values;
      h -> g (g in a later layer): h's change carried through g = m(h shifted) - m(h shifted, g held unedited).
    Per edge the median over items with its exact 95% interval -> DIR/dag_nat.json (natdraw draws it)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hs = json.load(open(os.path.expanduser(a.heads)))
    sa = a.select_alpha or hs["select_alpha"]
    H = [(u["layer"], u["index"]) for u in hs["heads"]]
    H.sort()
    model, tok = E.load(a.model, a.dtype, a.device)
    eng = E.Engine(model)
    items = json.load(open(os.path.expanduser(a.items)))
    items = items["items"] if isinstance(items, dict) else items
    dz = torch.load(os.path.expanduser(a.delta or hs["delta"]), weights_only=False)
    dl = {k: dz[k] for k in ("h", "n")}
    src = hs["source"]
    P = hs["P"]
    lab_ = [f"L{l}.H{i}" for l, i in H]
    mk = {h: masks(eng, [("h", *h)]) for h in H}
    mH = masks(eng, [("h", *h) for h in H])
    later = {h: masks(eng, [("h", *g) for g in H if g[0] > h[0]]) for h in H}
    pairs = [(i, j) for i in range(len(H)) for j in range(len(H)) if H[i][0] < H[j][0]]
    rec = {"P": P, "source": src, "alpha": sa, "heads": lab_, "delta": a.delta or hs["delta"], "families": {}}

    def dump():
        json.dump(rec, open(out / "dag_nat.json.tmp", "w"), indent=1)
        os.replace(out / "dag_nat.json.tmp", out / "dag_nat.json")
    for f in [f for f in a.families.split(",") if f]:
        splits = ["test"] if f in src else ["select", "test"]
        for lab, sign in [(x, 1.0 if x == "lie" else -1.0) for x in a.labels.split(",")]:
            rs = nat_recs(eng, tok, items, f, splits, lab, a.n_items, P)
            per, sents = {}, []
            for r in rs:
                seg, texts = segments(tok, r["ids"])
                c = sign * sa
                rows, keys = [(0.0, {}, None, None), (c, mH, None, None)], ["base", "H"]
                for g, pm in seg.items():
                    rows.append((c, mH, pm, None))
                    keys.append(f"src {g}")
                for hi, h in enumerate(H):
                    rows.append((c, mk[h], None, None))
                    keys.append(f"head {hi}")
                    rows.append((c, mk[h], None, later[h]))
                    keys.append(f"direct {hi}")
                    for g in GROUPS:
                        if g in seg:
                            rows.append((c, mk[h], seg[g], None))
                            keys.append(f"src {g} {hi}")
                for i, j in pairs:
                    rows.append((c, mk[H[i]], None, mk[H[j]]))
                    keys.append(f"via {i} {j}")
                clean = head_outputs(eng, r["ids"], {l for l, _ in H})
                lp = run_rows(eng, r["ids"], NatEdit(eng, dl, rows, clean))
                m = {k: eng.metric(row, r["lie"], r["truth"]) for k, row in zip(keys, lp)}
                del lp, clean
                tot = m["H"] - m["base"]
                if abs(tot) < 1e-6:
                    continue
                v = {"H": tot}
                for k, x in m.items():
                    if (k.startswith("src ") or k.startswith("head ") or k.startswith("direct ")) and \
                            not k[4:7] == "sys" or k.startswith("src system"):
                        v[k] = (x - m["base"]) / tot
                for i, j in pairs:
                    v[f"via {i} {j}"] = (m[f"head {i}"] - m[f"via {i} {j}"]) / tot
                for k, x in v.items():
                    per.setdefault(k, []).append(x)
                if texts:
                    sh = [round((m[f"src sys{q}"] - m["base"]) / tot, 4) for q in range(len(texts))]
                    sents.append({"id": r["it"]["id"], "shares": sh, "texts": [t[:300] for t in texts]})
            if not per:
                continue
            fam = {"n": len(per["H"]), "H_bits": median_ci(per["H"]),
                   "edges": {k: median_ci(x) + [len(x)] for k, x in per.items() if k != "H"},
                   "top_system_sentence": sents}
            rec["families"][f"{f} {lab}"] = fam
            src_line = "; ".join(f"{g} {fam['edges'][f'src {g}'][0]:+.2f}" for g in GROUPS
                                 if f"src {g}" in fam["edges"])
            top = sorted(range(len(H)), key=lambda i: -fam["edges"][f"head {i}"][0])[:6]
            E.log(f"natdag {f} {lab}: n {fam['n']} H moves m {fam['H_bits'][0]:+.2f} bits; source {src_line}; top heads "
                  + ", ".join(f"{lab_[i]} {fam['edges'][f'head {i}'][0]:+.2f} (direct "
                              f"{fam['edges'][f'direct {i}'][0]:+.2f})" for i in top) + f" ({time.time() - t0:.0f} s)")
            dump()
    E.log(f"natdag done {time.time() - t0:.0f} s -> {out / 'dag_nat.json'}")


def natablate(a):
    """The natsel head set switched off, no delta: each head's output held at zero, or at its mean over the source's
    select lies and resists (every position, pooled), at every position; against random same-layer head sets (natsel's
    seed) -> on each family's lies (removed: m < 0) and resists (induced: m > 0), and on the controls (share the
    unedited model gets right that stay right) -> DIR/ablate_nat.json."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hs = json.load(open(os.path.expanduser(a.heads)))
    core = [(u["layer"], u["index"]) for u in hs["heads"]]
    model, tok = E.load(a.model, a.dtype, a.device)
    eng = E.Engine(model)
    items = json.load(open(os.path.expanduser(a.items)))
    items = items["items"] if isinstance(items, dict) else items
    src, P = hs["source"], hs["P"]
    rng = np.random.default_rng(0)
    sets = [("set", core)]
    for q in range(a.random):
        rs = []
        for l, _i in core:
            free = [i for i in range(eng.nheads[l]) if (l, i) not in set(core) | set(rs)]
            rs.append((l, int(rng.choice(free))) if free else (l, _i))
        sets.append((f"random{q}", rs))
    layers = sorted({l for _, s in sets for l, _ in s})
    acc, n = {l: 0.0 for l in layers}, 0
    for lab in ("lie", "resist"):
        for r in [r for f in src for r in nat_recs(eng, tok, items, f, ["select"], lab, a.n_select, P)]:
            for l, x in head_outputs(eng, r["ids"], set(layers)).items():
                acc[l] = acc[l] + x.double().mean(0).cpu()
            n += 1
    fill = {"zero": {l: torch.zeros_like(acc[l]).float() for l in layers},
            "mean": {l: (acc[l] / max(n, 1)).float() for l in layers}}
    mks = [(nm, masks(eng, [("h", l, i) for l, i in s])) for nm, s in sets]
    rec = {"P": P, "source": src, "heads": [f"L{l}.H{i}" for l, i in core], "mean_items": n, "test": {}}

    def dump():
        json.dump(rec, open(out / "ablate_nat.json.tmp", "w"), indent=1)
        os.replace(out / "ablate_nat.json.tmp", out / "ablate_nat.json")

    def run(ids, how):
        rows = [(0.0, {}, None, None)] + [(0.0, {}, None, mk) for _, mk in mks]
        lp = run_rows(eng, ids, NatEdit(eng, {"h": [], "n": []}, rows, fill[how]))
        return lp[0], {nm: row for (nm, _), row in zip(mks, lp[1:])}
    for f in [f for f in a.families.split(",") if f]:
        splits = ["test"] if f in src else ["select", "test"]
        for lab in a.labels.split(","):
            rs = nat_recs(eng, tok, items, f, splits, lab, a.n_items, P)
            if not rs:
                continue
            tab = {}
            for how in ("zero", "mean"):
                per = {}
                for r in rs:
                    _, rows = run(r["ids"], how)
                    for nm, row in rows.items():
                        m = eng.metric(row, r["lie"], r["truth"])
                        per.setdefault(nm, []).append({"hit": (m < 0) if lab == "lie" else (m > 0),
                                                       "dm": (r["m"] - m) if lab == "lie" else (m - r["m"])})
                tab[how] = {nm: {"n": len(v), "hit": share_ci([float(x["hit"]) for x in v]),
                                 "dm": median_ci([x["dm"] for x in v])} for nm, v in per.items()}
                rnd = [tab[how][f"random{q}"]["hit"][0] for q in range(a.random)]
                s = tab[how]["set"]
                E.log(f"ablate {how} {f} {lab}: set {'removed' if lab == 'lie' else 'induced'} {s['hit'][0]:.2f} "
                      f"[{s['hit'][1]:.2f}, {s['hit'][2]:.2f}] dm {s['dm'][0]:+.2f}; random max {max(rnd):.2f} "
                      f"median {np.median(rnd):.2f} (n {s['n']}, {time.time() - t0:.0f} s)")
            rec["test"][f"{f} {lab}"] = tab
            dump()
    if a.controls:
        rec["controls"] = {}
        for g, crs in C.control_recs(tok, a.controls, a.ctl_groups.split(",")).items():
            stay, n_ = {}, 0
            for c in [c for c in crs if c["tail"] >= P][:a.n_items]:
                ids, truth = c["d"], c["it"]["truth_ids"]
                if int(eng.full_logp(ids).argmax()) not in truth:
                    continue
                n_ += 1
                for how in ("zero", "mean"):
                    _, rows = run(ids, how)
                    for nm, row in rows.items():
                        stay[(how, nm)] = stay.get((how, nm), 0) + int(int(row.argmax()) in truth)
            rec["controls"][g] = {"n": n_, **{how: {"set": round(stay.get((how, "set"), 0) / max(n_, 1), 4),
                                                    "random_mean": round(float(np.mean(
                                                        [stay.get((how, f"random{q}"), 0) / max(n_, 1)
                                                         for q in range(a.random)])), 4)}
                                              for how in ("zero", "mean")}}
            E.log(f"ablate controls {g}: {rec['controls'][g]}")
            dump()
    E.log(f"natablate done {time.time() - t0:.0f} s -> {out / 'ablate_nat.json'}")


def natpersona(a):
    """Anthropic's persona evals (model-written evals, persona/): per item m = log2 p(the answer matching the persona) -
    log2 p(the other answer), unedited and under +alpha x delta at every position on the natsel head set, on every
    head, and on the head set with --random random directions of the delta's per-head sizes (random_delta) ->
    DIR/persona.json, one record per item: m in that order (persona_stats.py tests it)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hs = json.load(open(os.path.expanduser(a.heads)))
    sa = a.select_alpha or hs["select_alpha"]
    model, tok = E.load(a.model, a.dtype, a.device)
    eng = E.Engine(model)
    dz = torch.load(os.path.expanduser(a.delta or hs["delta"]), weights_only=False)
    dl = {k: dz[k] for k in ("h", "n")}
    mH = masks(eng, [("h", u["layer"], u["index"]) for u in hs["heads"]])
    every = masks(eng, [("h", l, i) for l in range(eng.L) for i in range(eng.nheads[l])])
    rds = [random_delta(eng, dl, s) for s in range(1, a.random + 1)]
    rows = [(0.0, {}), (sa, mH), (sa, every)] + [(sa, mH)] * a.random
    dls = [dl, dl, dl] + rds
    names = ["base", "set", "every_head"] + [f"random{k}" for k in range(1, a.random + 1)]
    items = json.load(open(os.path.expanduser(a.items)))
    items = items["items"] if isinstance(items, dict) else items
    rec = {"heads": hs["heads"], "alpha": sa, "delta": a.delta or hs["delta"], "names": names, "items": []}
    for n, it in enumerate(items):
        ids, _ = E.item_seqs(tok, it)
        lp = run_rows(eng, ids, Shift(eng, dls, rows, 0, True))
        rec["items"].append({"id": it["id"], "trait": it["family"], "match": it["match"],
                             "m": [round(eng.metric(row, it["lie_ids"], it["truth_ids"]), 5) for row in lp]})
        if n % 100 == 99 or n == len(items) - 1:
            json.dump(rec, open(out / "persona.json.tmp", "w"))
            os.replace(out / "persona.json.tmp", out / "persona.json")
            x = np.array([r["m"] for r in rec["items"]])
            E.log(f"persona {n + 1}/{len(items)}: median m base {np.median(x[:, 0]):+.2f}, change set "
                  f"{np.median(x[:, 1] - x[:, 0]):+.2f}, every head {np.median(x[:, 2] - x[:, 0]):+.2f}, random "
                  f"{np.median(x[:, 3:] - x[:, :1]):+.2f} ({time.time() - t0:.0f} s)")
    E.log(f"natpersona done {time.time() - t0:.0f} s -> {out / 'persona.json'}")


@torch.no_grad()
def natgen(a):
    """Greedy replies (--max-new tokens, full forward per token) to --families' test lies, unsteered, steered on the
    natsel head set (+alpha x delta at every position, the reply's too) and steered on the set with one random direction
    of the delta's per-head sizes -> DIR/gen_nat.json."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hs = json.load(open(os.path.expanduser(a.heads)))
    sa = a.select_alpha or hs["select_alpha"]
    model, tok = E.load(a.model, a.dtype, a.device)
    eng = E.Engine(model)
    dz = torch.load(os.path.expanduser(a.delta or hs["delta"]), weights_only=False)
    dl = {k: dz[k] for k in ("h", "n")}
    mH = masks(eng, [("h", u["layer"], u["index"]) for u in hs["heads"]])
    conds = [("unsteered", dl, 0.0), ("steered", dl, sa), ("random direction", random_delta(eng, dl, 1), sa)]
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    items = json.load(open(os.path.expanduser(a.items)))
    items = items["items"] if isinstance(items, dict) else items
    rec = {"heads": hs["heads"], "alpha": sa, "replies": []}
    for f in [f for f in a.families.split(",") if f]:
        for r in nat_recs(eng, tok, items, f, ["test"], "lie", a.n_items, hs["P"]):
            row = {"id": r["it"]["id"], "family": f, "prompt": r["it"]["dec"], "prefill": r["it"].get("prefill", "")}
            for name, d, c in conds:
                ids, new = list(r["ids"]), []
                for _ in range(a.max_new):
                    lp = run_rows(eng, ids, Shift(eng, d, [(c, mH)], 0, True))[0]
                    t = int(lp.argmax())
                    if t in stop:
                        break
                    ids.append(t)
                    new.append(t)
                row[name] = tok.decode(new)
            rec["replies"].append(row)
            json.dump(rec, open(out / "gen_nat.json", "w"), indent=1)
            E.log(f"gen {row['id']}: unsteered {row['unsteered'][:80]!r} | steered {row['steered'][:80]!r} "
                  f"({time.time() - t0:.0f} s)")
    E.log(f"natgen done {time.time() - t0:.0f} s -> {out / 'gen_nat.json'}")


@torch.no_grad()
def natjlens(a):
    """J lens of the natsel heads' steering change: each head's slice of alpha x delta (and the whole set's) added at
    the decision position alone, read at that position through the rest of the network: (log2 p(+eps) - log2 p(-eps))
    / (2 eps) with the change scaled by eps, the Jacobian of the log-probs along it; averaged over contexts (the
    decision positions of --families' lies and resists, and of the controls) -> the tokens it raises and lowers most,
    over all contexts and per context group. Also: the controls' top tokens with the set switched off (zero at every
    position) against the unedited model's, and per item of --families the lie token, m unsteered and m steered
    (lies +alpha, resists -alpha) for the Yes/No polarity split -> DIR/jlens_nat.json."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hs = json.load(open(os.path.expanduser(a.heads)))
    sa, eps, P = a.select_alpha or hs["select_alpha"], a.eps, hs["P"]
    model, tok = E.load(a.model, a.dtype, a.device)
    eng = E.Engine(model)
    dz = torch.load(os.path.expanduser(a.delta or hs["delta"]), weights_only=False)
    dl = {k: dz[k] for k in ("h", "n")}
    units = [(f"L{u['layer']}.H{u['index']}", masks(eng, [("h", u["layer"], u["index"])])) for u in hs["heads"]]
    mH = masks(eng, [("h", u["layer"], u["index"]) for u in hs["heads"]])
    units.append(("set", mH))
    rows = [(s * eps * sa, mk) for _, mk in units for s in (1, -1)]
    items = json.load(open(os.path.expanduser(a.items)))
    items = items["items"] if isinstance(items, dict) else items
    ctx = []                                         # (group, ids)
    pol = []
    for f in [f for f in a.families.split(",") if f]:
        for lab in a.labels.split(","):
            for r in nat_recs(eng, tok, items, f, ["select", "test"], lab, a.n_items, P):
                ctx.append((f, r["ids"]))
                lp = run_rows(eng, r["ids"], Shift(eng, dl, [(sa if lab == "lie" else -sa, mH)], 0, True))[0]
                pol.append({"id": r["it"]["id"], "family": f, "label": lab, "lie_token": tok.decode(r["lie"][:1]),
                            "truth_token": tok.decode(r["truth"][:1]), "m": round(r["m"], 4),
                            "m_steered": round(eng.metric(lp, r["lie"], r["truth"]), 4)})
    ctl = C.control_recs(tok, a.controls, a.ctl_groups.split(",")) if a.controls else {}
    layers = sorted({u["layer"] for u in hs["heads"]})
    zero = {l: torch.zeros(eng.nheads[l] * eng.hdim[l]) for l in layers}
    off = []
    for g, crs in ctl.items():
        for c in crs[:a.n_ctl]:
            ctx.append(("controls", c["d"]))
            lp = run_rows(eng, c["d"], NatEdit(eng, {"h": [], "n": []}, [(0.0, {}, None, None), (0.0, {}, None, mH)],
                                               zero))
            off.append({"group": g, "truth": [tok.decode([t]) for t in sorted(c["it"]["truth_ids"])][:4],
                        **{k: [(tok.decode([int(t)]), round(float(v) / LN2, 2)) for v, t in zip(*torch.topk(lp[j], 5))]
                           for j, k in enumerate(("unedited", "switched_off"))}})
    E.log(f"jlens: {len(ctx)} contexts, {len(units)} units, eps {eps} x alpha {sa} ({time.time() - t0:.0f} s)")
    groups = sorted({g for g, _ in ctx})
    acc = {g: torch.zeros(len(units), eng.model.lm_head.weight.shape[0], dtype=torch.float64) for g in groups}
    cnt = {g: 0 for g in groups}
    per = max(1, a.jl_rows)
    for n, (g, ids) in enumerate(ctx):
        r = eng.prepare(ids, 1, [0], [0])
        r.drop_cap()
        lp = torch.cat([eng.span([r] * len(rows[b:b + per]), Shift(eng, dl, rows[b:b + per], 1, True))
                        for b in range(0, len(rows), per)])
        acc[g] += ((lp[0::2] - lp[1::2]) / (2 * eps * LN2)).double().cpu()
        cnt[g] += 1
        del r
        if n % 25 == 24:
            E.log(f"jlens {n + 1}/{len(ctx)} ({time.time() - t0:.0f} s)")

    def top(z, k):
        up, dn = torch.topk(z, k), torch.topk(-z, k)
        return {"up": [(tok.decode([int(t)]), round(float(s), 3)) for s, t in zip(up.values, up.indices)],
                "down": [(tok.decode([int(t)]), round(-float(s), 3)) for s, t in zip(dn.values, dn.indices)]}
    tot = sum(acc.values()) / max(1, sum(cnt.values()))
    rec = {"alpha": sa, "eps": eps, "contexts": cnt, "units": []}
    for j, (nm, _) in enumerate(units):
        rec["units"].append({"unit": nm, "all": top(tot[j], a.top), "norm": float(tot[j].norm()),
                             "groups": {g: top(acc[g][j] / max(1, cnt[g]), 10) for g in groups}})
        E.log(f"jlens {nm}: up {[w for w, _ in rec['units'][-1]['all']['up'][:10]]} down "
              f"{[w for w, _ in rec['units'][-1]['all']['down'][:10]]}")
    rec["switched_off"], rec["polarity"] = off, pol
    json.dump(rec, open(out / "jlens_nat.json", "w"), indent=1, ensure_ascii=False)
    E.log(f"natjlens done {time.time() - t0:.0f} s -> {out / 'jlens_nat.json'}")


@torch.no_grad()
def natneed(a):
    """Whether the natsel head set is needed for its delta's effect: +alpha x delta at every position (lies; -alpha for
    resists) on the set, on every head, and on every head but the set; per family the share of lies removed (m < 0) and
    of resists induced (m > 0) with exact intervals, and exact McNemar p of every head vs every head but the set ->
    DIR/need_nat.json."""
    from scipy.stats import binomtest
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hs = json.load(open(os.path.expanduser(a.heads)))
    sa, P = a.select_alpha or hs["select_alpha"], hs["P"]
    model, tok = E.load(a.model, a.dtype, a.device)
    eng = E.Engine(model)
    dz = torch.load(os.path.expanduser(a.delta or hs["delta"]), weights_only=False)
    dl = {k: dz[k] for k in ("h", "n")}
    core = {(u["layer"], u["index"]) for u in hs["heads"]}
    allh = [(l, i) for l in range(eng.L) for i in range(eng.nheads[l])]
    sets = [("set", masks(eng, [("h", *h) for h in core])), ("every_head", masks(eng, [("h", *h) for h in allh])),
            ("every_head_but_set", masks(eng, [("h", *h) for h in allh if h not in core]))]
    items = json.load(open(os.path.expanduser(a.items)))
    items = items["items"] if isinstance(items, dict) else items
    rec = {"alpha": sa, "families": {}}
    for f in [f for f in a.families.split(",") if f]:
        splits = ["test"] if f in hs["source"] else ["select", "test"]
        for lab in a.labels.split(","):
            rs = nat_recs(eng, tok, items, f, splits, lab, a.n_items, P)
            if not rs:
                continue
            c = sa if lab == "lie" else -sa
            hit = {nm: [] for nm, _ in sets}
            for r in rs:
                lp = run_rows(eng, r["ids"], Shift(eng, dl, [(c, mk) for _, mk in sets], 0, True))
                for (nm, _), row in zip(sets, lp):
                    m = eng.metric(row, r["lie"], r["truth"])
                    hit[nm].append((m < 0) if lab == "lie" else (m > 0))
            a_, b_ = (sum(x and not y for x, y in zip(hit["every_head"], hit["every_head_but_set"])),
                      sum(y and not x for x, y in zip(hit["every_head"], hit["every_head_but_set"])))
            p = binomtest(a_, a_ + b_, 0.5).pvalue if a_ + b_ else 1.0
            rec["families"][f"{f} {lab}"] = {"n": len(rs), **{nm: share_ci([float(x) for x in v])
                                                              for nm, v in hit.items()}, "mcnemar_p": p}
            E.log(f"need {f} {lab} (n {len(rs)}): " + "; ".join(f"{nm} {rec['families'][f'{f} {lab}'][nm][0]:.2f}"
                                                                for nm, _ in sets) +
                  f"; every head vs every head but the set McNemar p {p:.3g} ({time.time() - t0:.0f} s)")
            json.dump(rec, open(out / "need_nat.json", "w"), indent=1)
    E.log(f"natneed done {time.time() - t0:.0f} s -> {out / 'need_nat.json'}")


def natdraw(path: str, key: str):
    """DIR/dag_nat.json -> DIR/dag_nat_<family>_<label>.png: token groups -> heads (by layer) -> answer; an edge drawn
    where its exact 95% interval excludes zero, width by its median share of H's change."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rec = json.load(open(path))
    fam = rec["families"][key]
    E_, heads = fam["edges"], rec["heads"]
    sig = lambda k: k in E_ and (E_[k][1] > 0 or E_[k][2] < 0)
    layer = [int(h[1:].split(".")[0]) for h in heads]
    xs = sorted(set(layer))
    col = {l: 1 + i for i, l in enumerate(xs)}
    xans = len(xs) + 1.6
    pos, cnt = {}, {}
    for i, l in enumerate(layer):
        pos[i] = (col[l], cnt.get(l, 0))
        cnt[l] = cnt.get(l, 0) + 1
    hmax = max(cnt.values())
    pos = {i: (x, (y - (cnt[layer[i]] - 1) / 2)) for i, (x, y) in pos.items()}
    gs = [g for g in GROUPS if f"src {g}" in E_]
    gpos = {g: (-0.6, (k - (len(gs) - 1) / 2) * max(1.0, hmax / max(len(gs), 1))) for k, g in enumerate(gs)}
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(xs) + 4), max(5, 0.8 * hmax + 2.5)))
    drawn = [k for k in E_ if sig(k) and (k.startswith("via ") or k.startswith("direct ") or
                                      (k.startswith("src ") and k.count(" ") == 2))]
    wmax = max([abs(E_[k][0]) for k in drawn] + [1e-6])

    def edge(p, q, w, color):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=0.4 + 4 * abs(w) / wmax, color=color,
                                                        alpha=0.12 + 0.8 * abs(w) / wmax, shrinkA=7, shrinkB=7))
    pc, nc = "#2a6fdb", "#d1495b"
    for g in gs:
        for i in range(len(heads)):
            k = f"src {g} {i}"
            if sig(k):
                edge(gpos[g], pos[i], E_[k][0], pc if E_[k][0] > 0 else nc)
    for k in E_:
        if k.startswith("via ") and sig(k):
            i, j = map(int, k.split()[1:])
            edge(pos[i], pos[j], E_[k][0], pc if E_[k][0] > 0 else nc)
    for i in range(len(heads)):
        if sig(f"direct {i}"):
            edge(pos[i], (xans, 0), E_[f"direct {i}"][0], pc if E_[f"direct {i}"][0] > 0 else nc)
    for i, h in enumerate(heads):
        s = E_[f"head {i}"][0]
        ax.scatter(*pos[i], s=60 + 500 * min(abs(s), 1), color=pc if s > 0 else nc, zorder=3, edgecolor="white")
        ax.text(pos[i][0] + 0.12, pos[i][1] + 0.12, f"{h} {s:+.2f}", ha="left", va="bottom", fontsize=7)
    for g in gs:
        ax.scatter(*gpos[g], s=300, marker="s", color="#6c757d", zorder=3)
        ax.text(gpos[g][0] - 0.25, gpos[g][1], f"{g}\n{E_[f'src {g}'][0]:+.2f}", ha="right", va="center", fontsize=8)
    ax.scatter(xans, 0, s=400, marker="D", color="#222222", zorder=3)
    ax.text(xans + 0.25, 0, "answer\n(lie vs truth)", ha="left", va="center", fontsize=8)
    lo, hi = fam["H_bits"][1], fam["H_bits"][2]
    name = {"rp lie": "Role-play lies pushed to the truth", "rp resist": "Honest role-play replies pushed to the lie",
            "mask_yn lie": "MASK lies pushed to the truth", "mask_yn resist": "Honest MASK answers pushed to the lie"}
    ax.set_title(f"{name.get(key, key)}: shifting the {len(heads)} heads moves the answer {fam['H_bits'][0]:+.1f} bits "
                 f"[{lo:+.1f}, {hi:+.1f}] (median of {fam['n']} held-out items)", fontsize=10, pad=18)
    fig.text(0.01, 0.01, "Numbers: share of that move. Square -> head: the head shifted only at that part of the "
             "prompt. Head -> head: the part of the first head's effect lost when the second is held at its "
             "unedited output. Head -> answer: the head's effect with every later head held. Edges drawn where the "
             "exact 95% interval of the median excludes 0; width and opacity by size; red = toward the lie.",
             fontsize=7, color="#444444")
    ax.set_xticks([col[l] for l in xs] + [xans])
    ax.set_xticklabels([str(l) for l in xs] + [""], fontsize=8)
    ax.set_xlabel("layer")
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_xlim(-2.2, xans + 1.8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    png = Path(path).with_name(f"dag_nat_{key.replace(' ', '_')}.png")
    fig.savefig(png, dpi=150, facecolor="white")
    plt.close(fig)
    print(png)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "select", "natural", "natsel", "natdag", "natdraw", "natablate", "natpersona", "natgen", "natjlens", "natneed"])
    p.add_argument("--items", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--families", default="")
    p.add_argument("--heldout", default="")
    p.add_argument("--cores", default="")
    p.add_argument("--controls", default="")
    p.add_argument("--ctl-groups", default=C.CTL_GROUPS)
    p.add_argument("--alphas", default="0.5,1,2,4")
    p.add_argument("--span", type=int, default=9)
    p.add_argument("--n-select", type=int, default=64)
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--n-items", type=int, default=64)
    p.add_argument("--span-max", type=int, default=48)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ig-items", type=int, default=48)
    p.add_argument("--ig-steps", type=int, default=6)
    p.add_argument("--per-doubling", type=int, default=4)
    p.add_argument("--random", type=int, default=8)
    p.add_argument("--curve-items", type=int, default=32)
    p.add_argument("--select-alpha", type=float, default=2.0)
    p.add_argument("--wheres", default="every,span", help="run: delta at every position and/or the span only")
    p.add_argument("--project-out", default="",
                   help="run: remove from delta its component along the format-instruction direction of these control "
                        "groups (french,caps,json: each prompt minus the same prompt without the instruction)")
    p.add_argument("--random-deltas", type=int, default=0,
                   help="run: also K random deltas of the same per-unit sizes (every position)")
    p.add_argument("--source", default="", help="natural: the families whose select lie and resist items give delta")
    p.add_argument("--bins", type=int, default=4, help="natural: knowledge-margin bins delta is matched over")
    p.add_argument("--delta-at", default="decision", choices=["decision", "prompt"],
                   help="natural: delta's span: the P positions before the decision, or the prompt's last P "
                        "(before any reply prefix, which differs between lies and resists)")
    p.add_argument("--transfer-items", default="", help="natural: instructed items files (comma-separated)")
    p.add_argument("--transfer", default="", help="natural: instructed families scored as transfer")
    p.add_argument("--delta", default="", help="natsel: a natural delta (natural's delta_A_<source>.pt)")
    p.add_argument("--min-change", type=float, default=0.5,
                   help="natsel: items whose every-head m change is under this (bits) do not enter the ordering")
    p.add_argument("--heads", default="", help="natdag: natsel's heads_nat.json; natdraw: the dag_nat.json")
    p.add_argument("--labels", default="lie,resist", help="natdag: lie (+delta) and/or resist (-delta) items")
    p.add_argument("--key", default="", help="natdraw: the family and label to draw ('rp lie')")
    p.add_argument("--max-new", type=int, default=80, help="natgen: reply tokens")
    p.add_argument("--eps", type=float, default=0.25, help="natjlens: the change's scale for the Jacobian (x alpha)")
    p.add_argument("--n-ctl", type=int, default=8, help="natjlens: contexts per control group")
    p.add_argument("--jl-rows", type=int, default=20, help="natjlens: rows per span batch")
    p.add_argument("--top", type=int, default=25, help="natjlens: tokens listed per unit")
    if sys.argv[1:2] == ["natdraw"]:
        a, _ = p.parse_known_args(sys.argv[1:] + ["--items", "-", "--out", "-"])
        return natdraw(a.heads, a.key)
    a = p.parse_args()
    {"run": run, "select": select, "natural": natural, "natsel": natsel, "natdag": natdag, "natablate": natablate, "natpersona": natpersona, "natgen": natgen, "natjlens": natjlens, "natneed": natneed}[a.cmd](a)


if __name__ == "__main__":
    main()
