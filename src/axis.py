"""In-house Assistant Axis + persona displacement — the headline plot's x-axis.
Recipe: the AA contrast-vector construction — mean default activation minus the mean
of persona-mean activations — with each persona's mean response activation projected
onto it. Pure numpy over stored
activations — no GPU, no new generation.
  .venv/bin/python -m src.axis data/runs/<run_id>"""

import argparse
import json
from pathlib import Path

import numpy as np


def persona_means(run_dir):
    """Mean response_avg per persona over ALL that persona's responses — not the kept
    pairs. The grid is exactly balanced (equal pos/neg, equal splits) for every persona,
    so the trait-instruction contribution is identical across personas and cancels in the
    contrast; filtering first would unbalance it, since survival rates differ by persona."""
    z = np.load(run_dir / "activations.npz")
    persona_of = {r["response_id"]: r["persona"]
                  for r in map(json.loads, open(run_dir / "responses.jsonl"))}
    acc = {}
    for rid, row in zip(z["response_id"], z["response_avg"]):
        acc.setdefault(persona_of[str(rid)], []).append(row)
    return {p: np.stack(v).mean(axis=0) for p, v in sorted(acc.items())}, \
           {p: len(v) for p, v in acc.items()}


def build(run_dir):
    means, counts = persona_means(run_dir)
    assert "default" in means, "no default persona in this run — the axis is defined against it"
    others = [m for p, m in means.items() if p != "default"]
    # AA contrast recipe: default mean MINUS the mean of the (non-default) persona means.
    # FLAGGED: "persona means" is read as the non-default personas; including default would
    # dilute the contrast with itself. One-line change here if you read it the other way.
    axis = means["default"] - np.mean(others, axis=0)
    axis_hat = axis / np.linalg.norm(axis)
    disp = {p: float(m @ axis_hat) for p, m in means.items()}

    print(f"in-house Assistant Axis: ||axis|| = {np.linalg.norm(axis):.2f}, built from "
          f"default vs {len(others)} persona means")
    print(f"{'persona':<14}{'n_responses':>13}{'displacement':>14}")
    for p, d in sorted(disp.items(), key=lambda kv: -kv[1]):
        print(f"{p:<14}{counts[p]:>13}{d:>14.3f}")

    top = max(disp, key=disp.get)
    ok = top == "default"
    print(f"\nSELF-TEST {'PASS' if ok else 'FAIL'}: default sits at the {'max' if ok else 'NOT the'} "
          f"extreme (default {disp['default']:+.3f}, next {sorted(disp.values())[-2]:+.3f}, "
          f"min {min(disp.values()):+.3f})")
    assert ok, f"axis audit: default must sit at one extreme, but '{top}' does"

    np.savez(run_dir / "axis.npz", axis=axis, axis_hat=axis_hat,
             personas=np.array(sorted(means)), displacement=np.array([disp[p] for p in sorted(means)]))
    (run_dir / "axis.json").write_text(json.dumps(
        dict(displacement=disp, n_responses=counts, axis_norm=float(np.linalg.norm(axis)),
             recipe="default mean - mean of non-default persona means",
             scope="ALL responses per persona (balanced design), not kept pairs"), indent=2))
    print(f"-> {run_dir}/axis.npz, {run_dir}/axis.json  (metrics.py joins this as the plot x-axis)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    build(ap.parse_args().run_dir)
