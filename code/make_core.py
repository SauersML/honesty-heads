"""Writes the head set chosen by steer.py natsel as a core file for audit_eval.py: the heads, the span, and the shift
(the honesty direction at the selection's steering coefficient), the direction file given relative to the core file.
  python code/make_core.py HEADS.json DELTA.pt OUT.json"""
import json
import os
import sys

hs = json.load(open(sys.argv[1]))
out = {"P": hs["P"], "families": hs["source"], "select_items": [],
       "units": [{"kind": "head", "layer": u["layer"], "index": u["index"], "pb": 0} for u in hs["heads"]],
       "shift": {"alpha": hs["select_alpha"],
                 "delta_file": os.path.relpath(os.path.abspath(sys.argv[2]), os.path.dirname(os.path.abspath(sys.argv[3])))},
       "from_steer": sys.argv[1]}
json.dump(out, open(sys.argv[3], "w"))
