#!/usr/bin/env bash
# Reruns every step behind results/ on one GPU with at least 44 GB of memory (an A100 80GB: about 4 hours in all).
# Each step writes under $OUT (default out/); results/ holds the outputs of the original run.
#   bash run.sh
set -e
cd "$(dirname "$0")"
export PYTORCH_ALLOC_CONF=expandable_segments:True
OUT=${OUT:-out}
I=data/items_roleplay_insider.json
CTL=data/controls.json
mkdir -p "$OUT"

# 1. The honesty direction: per head, the mean output over the 9 tokens before the model's own truthful role-play
#    answers minus the same before its own role-play lies (the select split), with transfer tests.
python code/steer.py natural --items $I --source rp --families rp,it --out $OUT/natural --span 9 --alphas 1,2 \
  --wheres every,span --random-deltas 10 --bins 4 --n-select 400 --n-items 40 --controls $CTL \
  --ctl-groups honest_tf,honest_qa,french,caps,json,sys_caps,sys_negation,sys_fiction,sys_labelswap,sys_swap_shots,sys_honest

# 2. The heads: rank every head by integrated gradients, choose the fewest that reach 70% of the every-head effect on
#    lies and on truthful answers, and test them on the held-out role-play lies against random head sets.
python code/steer.py natsel --items $I --delta $OUT/natural/delta_A_rp.pt --source rp --families rp \
  --out $OUT/natsel_rp --span 9 --select-alpha 2 --ig-items 40 --ig-steps 6 --curve-items 40 --per-doubling 4 \
  --random 8 --n-items 40 --controls $CTL \
  --ctl-groups honest_tf,honest_qa,french,caps,json,sys_caps,sys_fiction,sys_labelswap,sys_swap_shots,sys_honest
H="--heads $OUT/natsel_rp/heads_nat.json --delta $OUT/natural/delta_A_rp.pt"

# 3. The circuit: path patching from prompt parts through the heads to the answer logit.
python code/steer.py natdag --items $I $H --families rp --labels lie,resist --n-items 40 --out $OUT/natdag_rp

# 4. Steering every head, and every head but the chosen ones (bfloat16).
python code/steer.py natneed --items $I $H --families rp --labels lie,resist --n-items 40 --dtype bfloat16 \
  --out $OUT/need

# 5. Switching the heads off, and greedy replies unsteered, steered and along a random direction.
python code/steer.py natablate --items $I $H --families rp --labels lie,resist --n-items 40 --n-select 64 \
  --random 8 --controls $CTL \
  --ctl-groups honest_tf,honest_qa,french,caps,sys_caps,sys_fiction,sys_labelswap,sys_swap_shots,sys_honest \
  --out $OUT/ablate
python code/steer.py natgen --items $I $H --families rp --n-items 6 --max-new 120 --out $OUT/gen

# 6. J lens of each head (what it writes toward the next token).
python code/steer.py natjlens --items $I $H --families rp --labels lie,resist --n-items 24 --controls $CTL \
  --ctl-groups honest_tf,honest_qa,french,caps,sys_caps,sys_fiction,sys_labelswap,sys_swap_shots,sys_honest \
  --out $OUT/jlens

# 7. Insider trading: full 512-token replies to the 45 conversations where the unsteered model lies to its manager,
#    steered and along 2 random directions, graded by rule.
mkdir -p $OUT/audit_it
python code/make_core.py $OUT/natsel_rp/heads_nat.json $OUT/natural/delta_A_rp.pt $OUT/audit_it/core_rp29.json
[ -e $OUT/audit_it/base.json ] || echo '{"lies": [], "controls": []}' > $OUT/audit_it/base.json
python code/audit_eval.py free --model Qwen/Qwen3.5-9B --items data/items_insider_audit.json --controls $CTL \
  --out $OUT/audit_it --dtype float32 --core $OUT/audit_it/core_rp29.json --free-items data/items_insider_audit.json \
  --families it --tag it --rand-free 2 --rand-modes every --free-max-new 512 --free-lies-only

# 8. Anthropic's persona evals: unsteered, steered, every head, and 39 random directions on the same heads.
python code/steer.py natpersona --items data/items_persona.json $H --random 39 --out $OUT/persona
python code/persona_stats.py $OUT/persona/persona.json $OUT/persona/persona_stats.json

# 9. The figure (from results/; point it at $OUT by copying the step outputs there).
python code/figure.py results/persona_stats.json honesty_heads.png

# 10. Other ways of steering toward honesty, each against its own random directions (persona_rep.py), and how their
#     persona changes compare with the 29 heads' (persona_compare.py). Neuron and residual strengths are calibrated on
#     the select lies to the every-head effect. (The arms in data/arms_*.json read the outputs under out/.)
R=$OUT/replication
python code/persona_rep.py resdir --items $I --source rp --out $R/residual
python code/persona_rep.py calib --items $I --arms data/arms_residual.json --ref-delta $OUT/natural/delta_A_rp.pt \
  --source rp --out $R/residual
python code/persona_rep.py run --items $I --persona data/items_persona.json --arms data/arms_residual.json \
  --calib $R/residual/calib.json --families rp --controls $CTL --out $R/residual
python code/persona_rep.py calib --items $I --arms data/arms_neurons.json --ref-delta $OUT/natural/delta_A_rp.pt \
  --source rp --out $R/neurons
SA=$(python -c "import json; print(json.load(open('$R/neurons/calib.json'))['arms']['every neuron']['alpha'])")
python code/steer.py natsel --kind n --items $I --delta $OUT/natural/delta_A_rp.pt --source rp --families rp \
  --out $R/neurons/natsel --span 9 --select-alpha $SA --ig-items 40 --ig-steps 6 --curve-items 40 --per-doubling 4 \
  --random 8 --n-items 40 --controls $CTL
python code/persona_rep.py run --items $I --persona data/items_persona.json --arms data/arms_neurons.json \
  --calib $R/neurons/calib.json --families rp --controls $CTL --out $R/neurons
python code/steer.py natsel --items $I --delta $OUT/natural/delta_A_rp.pt --source rp --families rp \
  --out $R/heads/natsel --span 9 --select-alpha 2 --ig-items 40 --ig-steps 0 --curve-items 40 --per-doubling 4 \
  --random 8 --n-items 40 --controls $CTL
python code/persona_rep.py run --items $I --persona data/items_persona.json --arms data/arms_heads.json \
  --families rp --controls $CTL --out $R/heads
python code/persona_compare.py $OUT/persona/persona.json $R/compare.json $R/residual $R/heads $R/neurons
python code/replication_figure.py results/replication/compare.json results/persona.json replication.png
python code/answer_split.py data/items_roleplay_insider.json $R/answer_split.json $R/heads/lies.json $R/neurons/lies.json \
  $R/residual/lies.json
python code/summary_figure/prepare.py
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --hide-scrollbars --force-device-scale-factor=2 \
  --window-size=2700,1985 --screenshot=summary.png "file://$PWD/code/summary_figure/figure.html"
