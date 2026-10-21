"""Does the persona result replicate under other ways of steering toward honesty? Each arm is one honesty steer built
from the model's own lies: a direction (steer.py natural's per-unit delta, or the residual stream's mean difference) on
a set of units (a natsel head or neuron set, every head, every MLP neuron, or the residual stream at one layer's input),
added at every position at strength alpha. Every arm is scored against its own random directions of the same per-unit
sizes on the same units (heads: random directions of each head's norm; neurons: each neuron's size with a random sign;
residual: a random direction of the same norm at the same layer), seeds 1..K as steer.py natpersona.

  persona_rep.py resdir --items NATURAL.json --source rp --out DIR [--span 9]: per layer input, the source's select
      resists' residual stream averaged over the span minus its select lies' (a plain difference: role-play items carry
      no knowledge margin) -> DIR/resid_<source>.pt
  persona_rep.py calib --items NATURAL.json --arms ARMS.json --out DIR: per arm, the alpha at which its median m change
      on the source's select lies (+alpha) and resists (-alpha), averaged, equals the reference's (every head along
      --ref-delta at alpha 2) -> DIR/calib.json
  persona_rep.py run --items NATURAL.json --persona PERSONA.json --arms ARMS.json --out DIR: per arm against its random
      directions, lie removal on the natural families' test lies (+alpha), lie induction on their test resists (-alpha)
      and the controls the unedited model gets right that stay right (+alpha) -> DIR/lies.json; then every persona item
      unsteered, under each arm and under its random directions -> DIR/persona.json (names: base, ARM, ARM/random1..K;
      test one arm with persona_stats.py DIR/persona.json OUT.json ARM)
ARMS.json: [{"name": ..., "delta": steer.py natural's .pt or resdir's .pt, "kind": "h" | "n" | "r", "units": "all" or
             natsel's heads_nat.json / neurons_nat.json, "exclude": a natsel file whose units are left out,
             "layer": l (kind r), "alpha": a number or "calib" (with
             "calib_from": another arm's calibrated alpha), "random": K}]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corefind as C  # noqa: E402
import engine as E  # noqa: E402
import steer as S  # noqa: E402

CTL = "honest_tf,honest_qa,french,caps,sys_caps,sys_fiction,sys_labelswap,sys_swap_shots,sys_honest"


def load_items(path: str) -> list:
    d = json.load(open(os.path.expanduser(path)))
    return d["items"] if isinstance(d, dict) else d


@torch.no_grad()
def resid_mean(eng, ids: list, P: int) -> torch.Tensor:
    """[L + 1, d] the residual stream at each layer's input (and after the last layer), averaged over the last P
    positions."""
    out = {}

    def bound(l, h):
        out[l] = h[0, -P:].double().mean(0).cpu()
        return h
    x = torch.as_tensor(ids, device=eng.dev)[None]
    eng._forward(x, torch.arange(len(ids), device=eng.dev)[None], eng._empty(1), None, False, boundary=bound)
    return torch.stack([out[l] for l in range(eng.L + 1)])


def resdir(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = E.load(a.model, a.dtype, a.device)
    eng = E.Engine(model)
    items, src, P = load_items(a.items), a.source.split(","), a.span
    mean = {}
    for lab in ("lie", "resist"):
        rs = [r for f in src for r in S.nat_recs(eng, tok, items, f, ["select"], lab, a.n_select, P)]
        mean[lab] = torch.stack([resid_mean(eng, r["ids"], P) for r in rs]).mean(0)
        E.log(f"resdir {lab}: {len(rs)} items")
    d = (mean["resist"] - mean["lie"]).float()
    stem = "resid_" + "+".join(src)
    torch.save({"r": list(d), "P": P, "source": src,
                "norm": d.norm(dim=1).tolist(), "resid_norm": mean["lie"].float().norm(dim=1).tolist()},
               out / f"{stem}.pt")
    E.log(f"resdir -> {out / stem}.pt; difference norm per layer {[round(x, 1) for x in d.norm(dim=1).tolist()]}")


def build(eng, arm: dict, alpha: float | None = None) -> tuple:
    """(rows [(coef, mask)], dls, names) of an arm and its random directions; coef the arm's alpha (or the one given)."""
    al = float(arm["alpha"]) if alpha is None else alpha
    dz = torch.load(os.path.expanduser(arm["delta"]), weights_only=False)
    K, kind = int(arm.get("random", 0)), arm["kind"]
    if kind == "r":
        l = int(arm["layer"])
        d = dz["r"][l].float()
        rd = []
        for s in range(1, K + 1):
            r = torch.randn(d.shape, generator=torch.Generator().manual_seed(s))
            rd.append({"r": {l: r / r.norm() * d.norm()}})
        dls, mk = [{"r": {l: d}}] + rd, {}
    else:
        dl = {k: dz[k] for k in ("h", "n")}
        dls = [dl] + [S.random_delta(eng, dl, s) for s in range(1, K + 1)]
        if arm["units"] == "all":
            us = [(kind, l, i) for l in range(eng.L) for i in range(eng.nheads[l] if kind == "h" else eng.D)]
        else:
            hs = json.load(open(os.path.expanduser(arm["units"])))
            us = [(kind, u["layer"], u["index"]) for u in hs["heads" if kind == "h" else "neurons"]]
        if arm.get("exclude"):
            hs = json.load(open(os.path.expanduser(arm["exclude"])))
            ex = {(kind, u["layer"], u["index"]) for u in hs["heads" if kind == "h" else "neurons"]}
            us = [u for u in us if u not in ex]
        mk = S.masks(eng, us)
    names = [arm["name"]] + [f"{arm['name']}/random{s}" for s in range(1, K + 1)]
    return [(al, mk)] * (K + 1), dls, names


def calib(a):
    """Alpha per arm on a log2 grid, interpolated in log alpha at the first grid step that reaches the reference's
    median m change (lies and resists averaged)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = E.load(a.model, a.dtype, a.device)
    eng = E.Engine(model)
    items, src, P = load_items(a.items), a.source.split(","), a.span
    arms = [x for x in json.load(open(a.arms)) if x.get("alpha") == "calib" and "calib_from" not in x]
    grid = [2.0 ** k for k in range(-3, 8)]
    ref = {"name": "reference", "delta": a.ref_delta, "kind": "h", "units": "all", "alpha": 2.0}
    plan = [(ref, 2.0)] + [(x, g) for x in arms for g in grid]
    rows, dls = [], []
    for arm, g in plan:
        r, d, _ = build(eng, {**arm, "random": 0}, g)
        rows += r
        dls += d
    dm = {lab: [] for lab in ("lie", "resist")}
    for lab, sign in (("lie", 1.0), ("resist", -1.0)):
        for r in [r for f in src for r in S.nat_recs(eng, tok, items, f, ["select"], lab, a.n_select, P)]:
            lp = S.run_rows(eng, r["ids"], S.Shift(eng, dls, [(sign * c, m) for c, m in rows], P, True))
            dm[lab].append([sign * (r["m"] - eng.metric(row, r["lie"], r["truth"])) for row in lp])
    med = {lab: np.median(np.array(v), 0) for lab, v in dm.items()}
    both = (med["lie"] + med["resist"]) / 2
    rec = {"reference": {"delta": a.ref_delta, "alpha": 2.0, "dm_lie": float(med["lie"][0]),
                         "dm_resist": float(med["resist"][0]), "dm": float(both[0])}, "grid": grid, "arms": {}}
    for j, arm in enumerate(arms):
        b = both[1 + j * len(grid): 1 + (j + 1) * len(grid)]
        k = int(np.argmax(b >= both[0])) if (b >= both[0]).any() else None
        if k is None:
            al = grid[int(np.argmax(b))]
        elif k == 0:
            al = grid[0]
        else:
            t = (both[0] - b[k - 1]) / (b[k] - b[k - 1])
            al = float(2.0 ** (np.log2(grid[k - 1]) + t * (np.log2(grid[k]) - np.log2(grid[k - 1]))))
        rec["arms"][arm["name"]] = {"alpha": al, "reached": k is not None, "dm": b.round(4).tolist(),
                                    "dm_lie": med["lie"][1 + j * len(grid): 1 + (j + 1) * len(grid)].round(4).tolist(),
                                    "dm_resist": med["resist"][1 + j * len(grid):
                                                               1 + (j + 1) * len(grid)].round(4).tolist()}
        E.log(f"calib {arm['name']}: alpha {al:.3g} (reached {k is not None}); m change per alpha "
              f"{dict(zip(grid, b.round(2).tolist()))}; reference {both[0]:.2f}")
    json.dump(rec, open(out / "calib.json", "w"), indent=1)


def resolve(arms: list, path: str) -> list:
    cal = json.load(open(path))["arms"] if path else {}
    return [{**x, "alpha": cal[x.get("calib_from", x["name"])]["alpha"]} if x.get("alpha") == "calib" else x
            for x in arms]


def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tok = E.load(a.model, a.dtype, a.device)
    eng = E.Engine(model)
    arms = resolve(json.load(open(a.arms)), a.calib)
    rows, dls, names = [(0.0, {})], [{}], ["base"]
    for arm in arms:
        r, d, n = build(eng, arm)
        rows, dls, names = rows + r, dls + d, names + n
    E.log(f"{len(arms)} arms, {len(names)} rows: " + "; ".join(f"{x['name']} alpha {float(x['alpha']):.3g}"
                                                               for x in arms))
    P = a.span
    lies = {"arms": arms, "names": names, "families": {}, "controls": {}}
    if (out / "lies.json").exists():                    # (resume: a finished lie and control pass is kept)
        old = json.load(open(out / "lies.json"))
        if old.get("done") and old["names"] == names:
            lies = old

    def dump_lies():
        json.dump(lies, open(out / "lies.json.tmp", "w"), indent=1)
        os.replace(out / "lies.json.tmp", out / "lies.json")
    items = load_items(a.items) if a.items and not lies.get("done") else []
    for f in [f for f in a.families.split(",") if f and items]:
        for lab, sign in (("lie", 1.0), ("resist", -1.0)):
            recs = S.nat_recs(eng, tok, items, f, ["test"], lab, a.n_items, P)
            if not recs:
                continue
            per = []
            for r in recs:
                lp = S.run_rows(eng, r["ids"], S.Shift(eng, dls[1:], [(sign * c, m) for c, m in rows[1:]], P, True))
                per.append([round(eng.metric(row, r["lie"], r["truth"]), 4) for row in lp])
            lies["families"][f"{f} {lab}"] = {"m": [round(r["m"], 4) for r in recs], "steered": per}
            x = np.array(per)
            hit = (x < 0) if sign > 0 else (x > 0)
            E.log(f"{f} {lab} ({len(recs)}): " + "; ".join(
                f"{n} {hit[:, j].mean():.2f}" for j, n in enumerate(names[1:]) if "/random" not in n) +
                "; random max " + ", ".join(
                f"{arm['name']} {max(hit[:, j].mean() for j, n in enumerate(names[1:]) if n.startswith(arm['name'] + '/random')):.2f}"
                for arm in arms if arm.get("random")))
            dump_lies()
    if a.controls and not lies.get("done"):
        for g, crs in C.control_recs(tok, a.controls, a.ctl_groups.split(",")).items():
            stay, n = np.zeros(len(names) - 1), 0
            for c in [c for c in crs if c["tail"] >= P][:a.n_ctl]:
                if int(eng.full_logp(c["d"]).argmax()) not in c["it"]["truth_ids"]:
                    continue
                n += 1
                lp = S.run_rows(eng, c["d"], S.Shift(eng, dls[1:], rows[1:], P, True))
                stay += np.array([int(row.argmax()) in c["it"]["truth_ids"] for row in lp])
            lies["controls"][g] = {"n": n, "stay": (stay / max(n, 1)).round(4).tolist()}
            E.log(f"controls {g} ({n}): " + "; ".join(f"{nm} {s:.2f}" for nm, s in zip(names[1:], stay / max(n, 1))
                                                       if "/random" not in nm))
            dump_lies()
    lies["done"] = True
    dump_lies()
    E.log(f"lies and controls done ({time.time() - t0:.0f} s)")
    rec = {"arms": arms, "names": names, "items": []}
    if (out / "persona.json").exists():                 # (resume from the items already scored)
        old = json.load(open(out / "persona.json"))
        if old["names"] == names:
            rec = old
    for n, it in enumerate(load_items(a.persona)):
        if n < len(rec["items"]):
            continue
        ids, _ = E.item_seqs(tok, it)
        lp = S.run_rows(eng, ids, S.Shift(eng, dls, rows, 0, True))
        rec["items"].append({"id": it["id"], "trait": it["family"], "match": it["match"],
                             "m": [round(eng.metric(row, it["lie_ids"], it["truth_ids"]), 5) for row in lp]})
        if n % 100 == 99:
            json.dump(rec, open(out / "persona.json.tmp", "w"))
            os.replace(out / "persona.json.tmp", out / "persona.json")
            x = np.array([r["m"] for r in rec["items"]])
            E.log(f"persona {n + 1}: " + "; ".join(f"{nm} {np.median(x[:, j] - x[:, 0]):+.2f}"
                                                   for j, nm in enumerate(names) if j and "/random" not in nm)
                  + f" ({time.time() - t0:.0f} s)")
    json.dump(rec, open(out / "persona.json.tmp", "w"))
    os.replace(out / "persona.json.tmp", out / "persona.json")
    E.log(f"persona_rep run done {time.time() - t0:.0f} s -> {out / 'persona.json'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["resdir", "calib", "run"])
    p.add_argument("--out", required=True)
    p.add_argument("--items", default="")
    p.add_argument("--persona", default="")
    p.add_argument("--arms", default="")
    p.add_argument("--calib", default="", help="run: calib.json for arms with alpha 'calib'")
    p.add_argument("--source", default="rp")
    p.add_argument("--families", default="rp,syco_v2,mask_yn")
    p.add_argument("--ref-delta", default="", help="calib: the reference delta (every head at alpha 2)")
    p.add_argument("--controls", default="")
    p.add_argument("--ctl-groups", default=CTL)
    p.add_argument("--span", type=int, default=9)
    p.add_argument("--n-select", type=int, default=64)
    p.add_argument("--n-items", type=int, default=40)
    p.add_argument("--n-ctl", type=int, default=16)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    {"resdir": resdir, "calib": calib, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
