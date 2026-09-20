"""Cosine between the in-house Assistant Axis and the sycophancy trait vector v_default,
both from the canonical grid run, read against the random-cosine floor in vectors_meta.json.
  .venv/bin/python scripts/axis_vs_vdefault.py"""
import json
import numpy as np

RUN = "data/runs/2026-09-03-grid-s0"
unit = lambda v: (v := v.astype(np.float64)) / np.linalg.norm(v)

cos = float(unit(np.load(f"{RUN}/axis.npz")["axis"]) @ unit(np.load(f"{RUN}/vectors.npz")["v_default"]))
meta = json.load(open(f"{RUN}/vectors_meta.json"))
print(f"cos(Assistant Axis, v_default) = {cos:+.4f}   (|cos| = {abs(cos):.4f})")
print(f"random-cosine floor            : mean {meta['random_cos_mean']:+.4f}, "
      f"abs max {meta['random_cos_absmax']:.4f}  (n=10 random unit vectors)")
