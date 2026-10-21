"""Deception common core (deccore, 10-20): the shared engine for Qwen3.5 hybrid decoders (Qwen/Qwen3.5-9B,
Qwen/Qwen3.8-27B), used by corefind.py (ordering, per-family circuits, the common core) and coretests.py (T1-T4).

Model: layer i is a Gated DeltaNet layer (linear_attn, out_proj) or gated full attention (self_attn, o_proj; i % 4 == 3);
every layer has a SwiGLU MLP. Units: a head at a position (its slice of the o_proj / out_proj input: 16 x 256 attention
heads or 32 x 128 DeltaNet value heads on 9B) and a neuron at a position (a down_proj input coordinate), keyed by
(kind, layer, index, pb) with pb the position counted back from the decision position (pb 0 predicts the answer).

Run: a sequence whose last P tokens are the span. Engine.prepare runs positions 0..s-1 once (s = T - P) and keeps each
layer's state at s (attention: keys after norm and rotary, values; DeltaNet: the last K-1 inputs of its causal
convolution and the recurrent state), then the span with the state, capturing every head output and neuron there.
Engine.span runs only the span positions of a batch of runs (rows of different prefix lengths: the attention prefix is
padded and masked; DeltaNet states are per row, so no padding enters a DeltaNet layer), with an edit applied to every
layer's head outputs and neurons. The engine computes each layer from the model's own modules (projections, norms,
the delta rule's torch implementation), so prefix + span equals the model's own forward to float error
(test_engine.py checks it, with the completeness identity: every head output at the span set to the honest run's
values gives the honest run's logits).

Metric at the last position: m = log2 sum p(lie ids) - log2 sum p(truth ids). N(S) = (m_dec - m_dec[S <- hon]) / gap,
R(S) = (m_hon[S <- dec] - m_hon) / gap, gap = m_dec - m_hon.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

LN2 = math.log(2.0)
BIG = 2 ** 30
BATCH_TOKENS = int(os.environ.get("DCORE_BATCH_TOKENS", 4096))   # span tokens a batch runs (rows x P)


def load(name: str, dtype: str = "float32", device: str = "cuda"):
    """(model, tokenizer): the text-only causal LM (AutoModelForCausalLM loads Qwen3_5ForCausalLM from the
    multimodal checkpoint), weights frozen."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False          # float32 means float32: no TF32 in matmuls or cuDNN
    torch.backends.cudnn.allow_tf32 = False                 # (the DeltaNet convolution)
    tok = AutoTokenizer.from_pretrained(name)
    kw = {"device_map": "balanced"} if device == "auto" else {}     # (auto: layers over every visible GPU)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype), **kw)
    if device != "auto":
        model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def prompt_ids(tok, msgs, prefill: str = "") -> list[int]:
    """measure.py's prompt: the chat template (thinking off) plus the prefill, tokenized whole."""
    s = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False) + prefill
    return tok.encode(s, add_special_tokens=False)


def shared_tail(a: list, b: list) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def item_seqs(tok, it: dict) -> tuple:
    """(dec ids, hon ids) of a measured item: each prompt plus the shared reply prefix before the decision (hon None for a
    deceptive-only item)."""
    pre = list(it.get("prefix_ids") or [])
    return (prompt_ids(tok, it["dec"], it.get("prefill", "")) + pre,
            prompt_ids(tok, it["hon"], it.get("prefill", "")) + pre if it.get("hon") else None)


def template_tail(tok, ids: list) -> int:
    """Tokens from the last <|im_end|> (the user turn's end) to the end: the chat template plus any reply prefix, the
    span a deceptive-only prompt shares with every other prompt."""
    e = tok.convert_tokens_to_ids("<|im_end|>")
    return len(ids) - max(i for i, t in enumerate(ids) if t == e)


def _delta_rule():
    """The model's chunked gated delta rule, its torch implementation (a hub/FLA kernel replaces it only when
    installed; the torch one computes in float32)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    f = mq.torch_chunk_gated_delta_rule
    return getattr(f, "__wrapped__", f)


# ---------------------------------------------------------------- units
class Units:
    """Flat ids of every unit at pb 0..P-1: per layer a heads block [P, nheads] then a neurons block [P, D]."""

    def __init__(self, eng: "Engine", P: int):
        self.P, self.blocks, off = P, {}, 0
        for l in range(eng.L):
            for kind, W in (("h", eng.nheads[l]), ("n", eng.D)):
                self.blocks[(kind, l)] = (off, W)
                off += P * W
        self.n = off
        self._starts = np.array([v[0] for v in self.blocks.values()])
        self._keys = list(self.blocks)

    def flat(self, kind: str, l: int, idx: int, pb: int) -> int:
        off, W = self.blocks[(kind, l)]
        return off + pb * W + idx

    def key(self, f: int) -> tuple:
        j = int(np.searchsorted(self._starts, f, side="right") - 1)
        kind, l = self._keys[j]
        off, W = self.blocks[(kind, l)]
        pb, idx = divmod(int(f) - off, W)
        return kind, l, idx, pb

    def keys(self, fs) -> list:
        return [self.key(f) for f in fs]

    def decode(self, fs) -> tuple:
        """Vectorized key: (block index into self._keys, idx, pb) arrays of flat ids."""
        fs = np.asarray(fs, dtype=np.int64)
        j = np.searchsorted(self._starts, fs, side="right") - 1
        W = np.array([self.blocks[k][1] for k in self._keys])[j]
        pb, idx = np.divmod(fs - self._starts[j], W)
        return j, idx, pb

    def view(self, vec: torch.Tensor, kind: str, l: int) -> torch.Tensor:
        """[P, W] view of a flat vector's block, rows in pb order (pb 0 first)."""
        off, W = self.blocks[(kind, l)]
        return vec[off:off + self.P * W].view(self.P, W)

    def to_span(self, vec: torch.Tensor, kind: str, l: int, P: int) -> torch.Tensor:
        """[P, W] of a flat vector's block in span order (the span's first position first) for a span of P <= self.P."""
        return self.view(vec, kind, l)[:P].flip(0)

    def from_keys(self, keys) -> np.ndarray:
        return np.array([self.flat(*k) for k in keys], dtype=np.int64)


def unit_json(k: tuple) -> dict:
    return {"kind": "head" if k[0] == "h" else "neuron", "layer": int(k[1]), "index": int(k[2]), "pb": int(k[3])}


# ---------------------------------------------------------------- runs
@dataclass
class Run:
    ids: list
    P: int
    lie: list
    truth: list
    s: int = 0
    kv: dict = field(default_factory=dict)       # attention layer -> (k [kvh, s, hd], v)
    conv: dict = field(default_factory=dict)     # DeltaNet layer -> [C, K-1]
    rec: dict = field(default_factory=dict)      # DeltaNet layer -> [H, dk, dv] float32
    cap: dict = field(default_factory=dict)      # "h"/"n" -> list over layers of [P, W] span values
    m: float = float("nan")
    top: int = -1
    lt: float = float("nan")     # log2 sum p(truth tokens)
    ll: float = float("nan")     # log2 sum p(lie tokens)

    def drop_cap(self):
        self.cap = {}


class Engine:
    def __init__(self, model):
        self.model, self.core = model, model.model
        self.layers = list(self.core.layers)
        cfg = model.config
        self.L, self.D = cfg.num_hidden_layers, cfg.intermediate_size
        self.full = [t == "full_attention" for t in cfg.layer_types]
        self.nheads, self.hdim = [], []
        for l, lay in enumerate(self.layers):
            if self.full[l]:
                self.nheads.append(cfg.num_attention_heads)
                self.hdim.append(lay.self_attn.head_dim)
            else:
                self.nheads.append(lay.linear_attn.num_v_heads)
                self.hdim.append(lay.linear_attn.head_v_dim)
        self.dev = self.core.embed_tokens.weight.device
        self.ldev = [lay.mlp.down_proj.weight.device for lay in self.layers]     # (device_map: layers over GPUs)
        self.dtype = self.core.embed_tokens.weight.dtype
        self.rule = _delta_rule()
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
        self._rope, self._repeat = mq.apply_rotary_pos_emb, mq.repeat_kv

    # -- layers
    def _attn(self, l, x, pos, kv0, kmask, edit):
        """x [B, P, d] normed input at positions pos [B, P]; kv0 (k, v) [B, kvh, S0, hd] prefix (S0 may be 0);
        kmask [B, S0] valid prefix keys. Returns (out [B, P, d], (k, v) of the span)."""
        a = self.layers[l].self_attn
        B, P, _ = x.shape
        hd = a.head_dim
        q, gate = torch.chunk(a.q_proj(x).view(B, P, -1, 2 * hd), 2, dim=-1)
        gate = gate.reshape(B, P, -1)
        q = a.q_norm(q).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(B, P, -1, hd)).transpose(1, 2)
        v = a.v_proj(x).view(B, P, -1, hd).transpose(1, 2)
        cos, sin = self.rope(x, pos)
        q, k = self._rope(q, k, cos, sin)
        K = torch.cat([kv0[0], k], 2) if kv0 is not None else k
        V = torch.cat([kv0[1], v], 2) if kv0 is not None else v
        S0 = K.shape[2] - P
        causal = torch.ones(P, P, dtype=torch.bool, device=x.device).tril()
        if S0:
            mask = torch.cat([kmask[:, None, :].expand(B, P, S0), causal[None].expand(B, -1, -1)], -1)[:, None]
        else:
            mask = causal[None, None].expand(B, 1, P, P)
        K, V = self._repeat(K, a.num_key_value_groups), self._repeat(V, a.num_key_value_groups)
        o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, scale=a.scaling)
        o = o.transpose(1, 2).reshape(B, P, -1) * torch.sigmoid(gate)
        if edit is not None:
            o = edit("h", l, o)
        return a.o_proj(o), (k, v)

    def _lin(self, l, x, conv0, rec0, edit):
        """x [B, P, d]; conv0 [B, C, K-1] the convolution's previous inputs; rec0 [B, H, dk, dv] or None.
        Returns (out, (conv state, recurrent state) after the span)."""
        g = self.layers[l].linear_attn
        B, P, _ = x.shape
        mixed = g.in_proj_qkv(x).transpose(1, 2)
        full = torch.cat([conv0, mixed], -1)
        c = F.silu(F.conv1d(full, g.conv1d.weight, g.conv1d.bias, groups=full.shape[1])).transpose(1, 2)
        q, k, v = torch.split(c, [g.key_dim, g.key_dim, g.value_dim], dim=-1)
        q = q.reshape(B, P, -1, g.head_k_dim)
        k = k.reshape(B, P, -1, g.head_k_dim)
        v = v.reshape(B, P, -1, g.head_v_dim)
        z = g.in_proj_z(x).reshape(B, P, -1, g.head_v_dim)
        beta = g.in_proj_b(x).sigmoid()
        dec = -g.A_log.float().exp() * F.softplus(g.in_proj_a(x).float() + g.dt_bias)
        if g.num_v_heads // g.num_k_heads > 1:
            q = q.repeat_interleave(g.num_v_heads // g.num_k_heads, dim=2)
            k = k.repeat_interleave(g.num_v_heads // g.num_k_heads, dim=2)
        cs = 64 if P > 64 else max(8, 1 << (P - 1).bit_length())     # (a short span padded to its own chunk, not 64)
        o, rec = self.rule(q, k, v, g=dec, beta=beta, chunk_size=cs, initial_state=rec0, output_final_state=True,
                           use_qk_l2norm_in_kernel=True)
        o = g.norm(o.reshape(-1, g.head_v_dim), z.reshape(-1, g.head_v_dim)).reshape(B, P, -1)
        if edit is not None:
            o = edit("h", l, o)
        return g.out_proj(o), (full[:, :, full.shape[-1] - (g.conv_kernel_size - 1):], rec)

    def rope(self, x, pos):
        """(cos, sin) of the rotary embedding at positions pos [B, P], on x's device."""
        rd = self.core.rotary_emb.inv_freq.device
        cos, sin = self.core.rotary_emb(x.to(rd), pos.to(rd)[None].expand(3, -1, -1))
        return cos.to(x.device), sin.to(x.device)

    def _mlp(self, l, h, edit):
        lay = self.layers[l]
        y = lay.post_attention_layernorm(h)
        mlp = lay.mlp
        act = mlp.act_fn(mlp.gate_proj(y)) * mlp.up_proj(y)
        if edit is not None:
            act = edit("n", l, act)
        return h + mlp.down_proj(act)

    def _forward(self, ids, pos, state, edit, keep: bool, *, initial=None, start_layer=0, boundary=None):
        """Positions given by ids [B, P] at pos [B, P]; state(l): layer l's batched (kv0, kmask) or (conv0, rec0),
        built when the layer runs. Returns (final hidden at the last position [B, d], new states per layer when keep)."""
        # Optional residual boundaries support causal suffix interventions without duplicating the engine.
        h = self.core.embed_tokens(ids) if initial is None else initial
        new = {}
        for l in range(start_layer, self.L):
            lay = self.layers[l]
            if boundary is not None:
                h = boundary(l, h)
            if h.device != self.ldev[l]:
                h, pos = h.to(self.ldev[l]), pos.to(self.ldev[l])
            x = lay.input_layernorm(h)
            if self.full[l]:
                kv0, km = state(l)
                o, st = self._attn(l, x, pos, kv0, km, edit)
            else:
                conv0, rec0 = state(l)
                o, st = self._lin(l, x, conv0, rec0, edit)
            if keep:
                new[l] = st
            h = self._mlp(l, h + o, edit)
        if boundary is not None:
            h = boundary(self.L, h)
        return self.core.norm(h[:, -1].to(self.core.norm.weight.device)), new

    def _empty(self, B: int):
        def get(l):
            if self.full[l]:
                return None, None
            g = self.layers[l].linear_attn
            return torch.zeros(B, g.conv_dim, g.conv_kernel_size - 1, dtype=self.dtype, device=self.ldev[l]), None
        return get

    def logp(self, hn: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.model.lm_head(hn.to(self.model.lm_head.weight.device)).float(), -1)

    @staticmethod
    def metric(lp: torch.Tensor, lie: list, truth: list) -> float:
        return float((torch.logsumexp(lp[lie], 0) - torch.logsumexp(lp[truth], 0)) / LN2)

    # -- runs
    @torch.no_grad()
    def full_logp(self, ids: list) -> torch.Tensor:
        """[V] log-probs after the whole sequence by the engine's layers in one pass (no span split)."""
        x = torch.as_tensor(ids, device=self.dev)[None]
        pos = torch.arange(x.shape[1], device=self.dev)[None]
        hn, _ = self._forward(x, pos, self._empty(1), None, False)
        return self.logp(hn)[0]

    @torch.no_grad()
    def prepare(self, ids: list, P: int, lie: list, truth: list) -> Run:
        """The run's prefix state at s = T - P and its span captures and metric."""
        T = len(ids)
        r = Run(ids=list(ids), P=P, lie=list(lie), truth=list(truth), s=T - P)
        st = self._empty(1)
        if r.s > 0:
            x = torch.as_tensor(ids[:r.s], device=self.dev)[None]
            pos = torch.arange(r.s, device=self.dev)[None]
            _, new = self._forward(x, pos, st, None, True)
            for l in range(self.L):
                if self.full[l]:
                    r.kv[l] = (new[l][0][0], new[l][1][0])
                else:
                    r.conv[l], r.rec[l] = new[l][0][0].clone(), new[l][1][0].clone()   # (a slice: clone, not the prefix's inputs)
        else:
            for l in range(self.L):
                if self.full[l]:
                    a = self.layers[l].self_attn
                    kvh = a.config.num_key_value_heads
                    z = torch.zeros(kvh, 0, a.head_dim, dtype=self.dtype, device=self.ldev[l])
                    r.kv[l] = (z, z)
                else:
                    r.conv[l], r.rec[l] = st(l)[0][0], None
        cap = {"h": [None] * self.L, "n": [None] * self.L}

        def grab(kind, l, x):
            cap[kind][l] = x[0].detach().clone()
            return x
        lp = self.span([r], grab)[0]
        r.cap = cap
        r.m = self.metric(lp, r.lie, r.truth)
        r.top = int(lp.argmax())
        r.lt, r.ll = (float(torch.logsumexp(lp[x], 0) / LN2) for x in (r.truth, r.lie))
        return r

    def _state(self, runs: list):
        """state(l): batched prefix states of runs (one P), built per layer from the batch's distinct runs: attention
        prefixes padded to the longest and masked."""
        uniq, idx = {}, []
        for r in runs:
            idx.append(uniq.setdefault(id(r), (len(uniq), r))[0])
        us = [v[1] for v in sorted(uniq.values(), key=lambda v: v[0])]
        ix = torch.tensor(idx, device=self.dev)
        S0 = max(r.s for r in us)

        def get(l):
            if self.full[l]:
                if S0 == 0:
                    return None, None
                k0 = us[0].kv[l][0]
                K = torch.zeros(len(us), k0.shape[0], S0, k0.shape[2], dtype=k0.dtype, device=k0.device)
                V = torch.zeros_like(K)
                km = torch.zeros(len(us), S0, dtype=torch.bool, device=k0.device)
                for b, r in enumerate(us):
                    if r.s:
                        K[b, :, S0 - r.s:] = r.kv[l][0]
                        V[b, :, S0 - r.s:] = r.kv[l][1]
                        km[b, S0 - r.s:] = True
                i_ = ix.to(K.device)
                return (K[i_], V[i_]), km[i_]
            conv = torch.stack([r.conv[l] for r in us])
            conv = conv[ix.to(conv.device)]
            recs = [r.rec[l] for r in us]
            if all(x is None for x in recs):
                return conv, None
            ref = next(x for x in recs if x is not None)
            return conv, torch.stack([x if x is not None else torch.zeros_like(ref) for x in recs])[ix.to(ref.device)]
        return get

    def span(self, runs: list, edit=None) -> torch.Tensor:
        """[B, V] float32 log-probs at the last position of each run, only the span positions computed (one batch;
        all runs have the same P). edit(kind, l, x [B, P, W]) -> x is applied to head outputs and neurons."""
        P = runs[0].P
        assert all(r.P == P for r in runs)
        ids = torch.tensor([r.ids[r.s:] for r in runs], device=self.dev)
        pos = torch.tensor([list(range(r.s, r.s + P)) for r in runs], device=self.dev)
        hn, _ = self._forward(ids, pos, self._state(runs), edit, False)
        return self.logp(hn)


# ---------------------------------------------------------------- patching
class Patch:
    """An edit for Engine.span over a batch of rows: row b sets the units of its set (at the span) to the values of its
    source capture. A set is ("rank", R, k): the units whose rank in R (a flat int32 vector over Units) is below k, or
    ("units", flat ids). Sources: captures {"h"/"n": [per layer [P, W]]} (a Run's cap, another item's, a mean)."""

    def __init__(self, eng: Engine, U: Units, P: int, sets: list, srcs: list):
        self.eng, self.U, self.P, self.sets, self.srcs = eng, U, P, sets, srcs
        self.B = len(sets)
        dev = eng.dev
        self._k = torch.tensor([s[2] if s[0] == "rank" else 0 for s in sets], device=dev)
        self._sparse = {}                     # (kind, l) -> (b, p, idx) index tensors of the "units" rows
        rows = [(b, np.asarray(s[1], dtype=np.int64)) for b, s in enumerate(sets) if s[0] == "units" and len(s[1])]
        if rows:
            fl = np.concatenate([f for _, f in rows])
            bb = np.concatenate([np.full(len(f), b) for b, f in rows])
            j, idx, pb = U.decode(fl)
            ok = pb < P
            for jj in np.unique(j[ok]):
                sel = ok & (j == jj)
                t = torch.as_tensor(np.stack([bb[sel], P - 1 - pb[sel], idx[sel]]), device=dev)
                self._sparse[U._keys[jj]] = (t[0], t[1], t[2])
        self._ranks = [s[1] for s in sets]

    def mask(self, kind: str, l: int) -> torch.Tensor | None:
        W = self.eng.nheads[l] if kind == "h" else self.eng.D
        m = None
        rk = [b for b, s in enumerate(self.sets) if s[0] == "rank" and s[2] > 0]
        if rk:
            m = torch.zeros(self.B, self.P, W, dtype=torch.bool, device=self.eng.dev)
            uniq = {}
            for b in rk:
                uniq.setdefault(id(self._ranks[b]), (self._ranks[b], []))[1].append(b)
            for R, bs in uniq.values():
                r = self.U.to_span(R, kind, l, self.P)
                bi = torch.tensor(bs, device=self.eng.dev)
                m[bi] = r[None] < self._k[bi][:, None, None]
        sp = self._sparse.get((kind, l))
        if sp is not None:
            if m is None:
                m = torch.zeros(self.B, self.P, W, dtype=torch.bool, device=self.eng.dev)
            m[sp] = True
        return m

    def __call__(self, kind: str, l: int, x: torch.Tensor) -> torch.Tensor:
        m = self.mask(kind, l)
        if m is None or not bool(m.any()):
            return x
        m = m.to(x.device)
        if kind == "h":
            m = m.repeat_interleave(self.eng.hdim[l], dim=-1)
        src = torch.stack([s[kind][l].to(x.device) for s in self.srcs])
        return torch.where(m, src.to(device=x.device, dtype=x.dtype), x)


def rank_of(order: np.ndarray, n: int, dev) -> torch.Tensor:
    """Flat int32 ranks of an order of flat ids (units outside it: BIG)."""
    R = torch.full((n,), BIG, dtype=torch.int32)
    R[torch.as_tensor(np.asarray(order, dtype=np.int64))] = torch.arange(len(order), dtype=torch.int32)
    return R.to(dev)


@dataclass
class Pair:
    item: dict
    dec: Run
    hon: Run | None           # None: a deceptive-only item (no honest version), or a control prompt

    @property
    def gap(self) -> float:
        return self.dec.m - self.hon.m if self.hon is not None else float("nan")


TRUTH_MIN = 0.1      # bits: below this honest-minus-deceptive truth mass, Nt is undefined


def shares(eng: Engine, U: Units, pairs: list, sets: list, src=None, batch_tokens: int | None = None) -> dict:
    """Measured N and R of each set on each pair: {"N": [pairs, sets], "R": ..., "flipN": bool (the patched dec run's top
    token is a truth id), "flipR" (the patched hon run's top token is a lie id), "mN", "mR", "Nt" (share of log2 p(truth)
    recovered), "Rl" (share of log2 p(lie) induced), "Nc" (credited N: min(N, Nt)), "Rc" (credited R: min(R, Rl))} as
    numpy arrays.
    src(pair index, "N"|"R") -> capture overrides the partner's capture as the source (cross-item, means); "R" may
    return None to skip sufficiency. A pair without an honest run gets "N" only, as the patched m and flip (N nan), and
    nothing without src (no partner)."""
    P = pairs[0].dec.P
    conv = {}                                 # large unit sets as rank vectors (a compare, not a scatter, per batch)
    for s in sets:
        if s[0] == "units" and len(s[1]) > 64 and id(s[1]) not in conv:
            conv[id(s[1])] = ("rank", rank_of(s[1], U.n, eng.dev), len(s[1]))
    sets = [conv.get(id(s[1]), s) if s[0] == "units" else s for s in sets]
    jobs = []
    for i, pr in enumerate(pairs):
        for j, s in enumerate(sets):
            for d in (("N", "R") if pr.hon is not None else ("N",)):
                c = src(i, d) if src is not None else (None if pr.hon is None else pr.hon.cap if d == "N" else pr.dec.cap)
                if c is None:
                    continue
                jobs.append((i, j, d, pr.dec if d == "N" else pr.hon, c, s))
    out = {k: np.full((len(pairs), len(sets)), np.nan) for k in ("N", "R", "mN", "mR", "Nt", "Rl")}
    flips = {k: np.zeros((len(pairs), len(sets)), dtype=bool) for k in ("flipN", "flipR")}
    per = max(1, (batch_tokens or BATCH_TOKENS) // P)
    with torch.no_grad():
        for a in range(0, len(jobs), per):
            chunk = jobs[a:a + per]
            pt = Patch(eng, U, P, [c[5] for c in chunk], [c[4] for c in chunk])
            lp = eng.span([c[3] for c in chunk], pt)
            top = lp.argmax(-1).tolist()
            for (i, j, d, base, _, _), row, t in zip(chunk, lp, top):
                pr = pairs[i]
                m = eng.metric(row, base.lie, base.truth)
                if d == "N":
                    out["N"][i, j], out["mN"][i, j] = (pr.dec.m - m) / pr.gap, m
                    flips["flipN"][i, j] = t in pr.dec.truth
                    if pr.hon is not None and pr.hon.lt - pr.dec.lt > TRUTH_MIN:
                        out["Nt"][i, j] = (float(torch.logsumexp(row[base.truth], 0) / LN2) - pr.dec.lt) / (pr.hon.lt - pr.dec.lt)
                else:
                    out["R"][i, j], out["mR"][i, j] = (m - pr.hon.m) / pr.gap, m
                    flips["flipR"][i, j] = t in pr.hon.lie
                    if pr.dec.ll - pr.hon.ll > TRUTH_MIN:
                        out["Rl"][i, j] = (float(torch.logsumexp(row[base.lie], 0) / LN2) - pr.hon.ll) / (pr.dec.ll - pr.hon.ll)
    out.update(flips)
    # credited N: the lie removed only as far as this item's truth comes back (min of the share of m and of log2 p(truth)
    # recovered; m alone where the honest run gives the truth no more mass than the deceptive one)
    out["Nc"] = np.where(np.isfinite(out["Nt"]), np.fmin(out["N"], out["Nt"]), out["N"])
    # credited R, the same way: the lie induced only as far as this item's lie tokens gain log2 p (an edit that only
    # erases the honest answer, moving m toward 0, earns nothing)
    out["Rc"] = np.where(np.isfinite(out["Rl"]), np.fmin(out["R"], out["Rl"]), out["R"])
    out["mdec"] = np.array([pr.dec.m for pr in pairs])     # [pairs]: the unpatched deceptive margin
    return out


def boot_median(x, n: int = 2000, seed: int = 0) -> tuple:
    """(median, 2.5%, 97.5%) of a median over items by a percentile bootstrap."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    b = np.median(x[rng.integers(0, len(x), (n, len(x)))], 1)
    return float(np.median(x)), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def boot_mean(x, n: int = 2000, seed: int = 0) -> tuple:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def grid(n: int, per_doubling: int = 8) -> list[int]:
    """circuit_finder.grid's prefix sizes: every size to 16, then per_doubling per doubling, and n."""
    ks, k = list(range(1, min(n, 16) + 1)), 16
    while k < n:
        k = int(math.ceil(k * 2 ** (1 / per_doubling)))
        ks.append(min(k, n))
    return sorted(set(ks))


def log(*a):
    print(*a, flush=True)
    sys.stdout.flush()
