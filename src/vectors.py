"""Trait-vector extraction: difference-of-means on response_avg over KEPT extract-split
pairs. Pure numpy over activations.npz — no GPU, no model.
Kept-pair logic is imported from filter_report, never re-implemented.
  .venv/bin/python -m src.vectors --self-test              # synthetic fixture, no data needed
  .venv/bin/python -m src.vectors data/runs/<run_id>       # writes vectors.npz
  .venv/bin/python -m src.vectors data/runs/<run_id> --no-filter  # filter-bias check"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.config import SEED
from src.filter_report import kept_pairs

N_RANDOM = 10  # random unit vectors for the cosine floor (pre-registered baseline 2)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def diff_of_means(acts, pairs):
    """v = mean(pos activations) - mean(neg activations) over the given kept pairs."""
    pos = np.stack([acts[p["pos"]["response_id"]] for _, p in pairs])
    neg = np.stack([acts[p["neg"]["response_id"]] for _, p in pairs])
    return pos.mean(axis=0) - neg.mean(axis=0), len(pairs)


def extract(run_dir, use_filter=True):
    z = np.load(run_dir / "activations.npz")
    acts = {site: {rid: row for rid, row in zip(z["response_id"], z[site])}
            for site in ("response_avg", "prompt_last")}
    rng = np.random.default_rng(SEED)
    vectors, report, ceilings = {}, [], {}
    tag = "" if use_filter else "_nofilter"
    personas = sorted({k[0] for k, _ in kept_pairs(run_dir, split="extract", use_filter=use_filter)})

    for persona in personas:
        pairs = kept_pairs(run_dir, persona=persona, split="extract", use_filter=use_filter)
        v, n = diff_of_means(acts["response_avg"], pairs)
        vectors[f"v_{persona}"] = v
        # (7) same kept pairs, prompt_last site: if this is ~collinear with the response_avg
        # vector, the "trait vector" is substantially an instruction detector, not a
        # readout of expressed behaviour
        v_pl, _ = diff_of_means(acts["prompt_last"], pairs)
        vectors[f"pl_{persona}"] = v_pl
        # (6) split-half ceiling for EVERY extraction persona, split BY PAIR
        idx = rng.permutation(len(pairs))
        va, _ = diff_of_means(acts["response_avg"], [pairs[i] for i in idx[: len(idx) // 2]])
        vb, _ = diff_of_means(acts["response_avg"], [pairs[i] for i in idx[len(idx) // 2:]])
        ceilings[persona] = cos(va, vb)
        vectors[f"ceiling_{persona}"] = np.array(ceilings[persona])
        report.append((persona, n, float(np.linalg.norm(v)), cos(v, v_pl)))
        if persona == "default":
            vectors["v_default_half_a"], vectors["v_default_half_b"] = va, vb

    v_def = vectors["v_default"]
    D_MODEL = v_def.shape[0]  # from the data, not config — the 32B run is d=5120
    rand = rng.normal(size=(N_RANDOM, D_MODEL))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    rand_cos = np.array([cos(v_def, r) for r in rand])
    vectors["random_cosines"] = rand_cos
    vectors["split_half_cosine"] = np.array(ceilings["default"])  # kept name: default's ceiling
    vectors["personas"] = np.array(personas)

    scope = "KEPT pairs (paired filter)" if use_filter else "ALL raw pairs (NO filter — bias check)"
    print(f"extraction scope: {scope}\n")
    print(f"{'vector':<16}{'n_pairs':>9}{'norm':>9}{'cos vs v_default':>18}{'own ceiling':>13}"
          f"{'cos(resp_avg, prompt_last)':>28}")
    for persona, n, norm, cos_pl in report:
        print(f"{'v_' + persona:<16}{n:>9}{norm:>9.1f}{cos(vectors['v_' + persona], v_def):>18.4f}"
              f"{ceilings[persona]:>13.4f}{cos_pl:>28.4f}")
    print("\n(own ceiling = that persona's own split-half cosine — its extraction reliability)")
    print("(cos(resp_avg, prompt_last): high => the direction is largely an INSTRUCTION "
          "detector readable before the response exists, not a readout of expressed behaviour)")

    print(f"\nrotation references for each persona vs default:")
    print(f"{'persona':<14}{'cos vs default':>16}{'default ceiling':>18}{'corrected ref':>16}")
    for persona in personas:
        if persona == "default":
            continue
        c = cos(vectors[f"v_{persona}"], v_def)
        corrected = math.sqrt(max(0.0, ceilings["default"] * ceilings[persona]))
        print(f"{persona:<14}{c:>16.4f}{ceilings['default']:>18.4f}{corrected:>16.4f}")
    print("(corrected ref = sqrt(ceil_default * ceil_persona): attenuation applies to BOTH "
          "vectors, so this is the achievable ceiling for the PAIR. See config.py.)")
    print(f"random floor ({N_RANDOM} unit vectors): mean {rand_cos.mean():+.4f}, "
          f"max |cos| {np.abs(rand_cos).max():.4f}  (expected ~0 +- 0.017 at d={D_MODEL})")

    np.savez(run_dir / f"vectors{tag}.npz", **vectors)
    (run_dir / f"vectors_meta{tag}.json").write_text(json.dumps(dict(
        site="response_avg", split="extract", use_filter=use_filter,
        n_pairs={p: n for p, n, _, _ in report}, ceilings=ceilings,
        split_half_cosine=ceilings["default"],
        corrected_refs={p: math.sqrt(max(0.0, ceilings["default"] * ceilings[p]))
                        for p in personas if p != "default"},
        cos_response_avg_vs_prompt_last={p: c for p, _, _, c in report},
        random_cos_mean=float(rand_cos.mean()), random_cos_absmax=float(np.abs(rand_cos).max()),
        stored="raw (non-unit) difference-of-means; metrics.py unit-normalizes"), indent=2))
    print(f"\n-> {run_dir}/vectors{tag}.npz")
    return vectors


def self_test():
    """Toy-walkthrough fixture: plant a direction, add nuisance offset + noise, check
    difference-of-means recovers it and that split-half tracks the sampling ceiling."""
    rng = np.random.default_rng(0)
    d, n, signal, noise = 256, 60, 4.0, 0.3
    # signal must beat the noise NORM, which grows as sqrt(d): ||noise_diff|| ~
    # noise*sqrt(2/n)*sqrt(d) = 0.88 here vs signal 4.0 -> cos ~ 0.98
    true_dir = rng.normal(size=d)
    true_dir /= np.linalg.norm(true_dir)
    base = rng.normal(size=d) * 50         # huge persona offset, common to both polarities:
                                           # it must cancel exactly in the difference (offset trap)
    pos = base + signal * true_dir + rng.normal(size=(n, d)) * noise
    neg = base + rng.normal(size=(n, d)) * noise

    acts, pairs = {}, []
    for i in range(n):
        acts[f"p{i}"], acts[f"n{i}"] = pos[i], neg[i]
        pairs.append(((None,), {"pos": {"response_id": f"p{i}"}, "neg": {"response_id": f"n{i}"}}))
    v, got_n = diff_of_means(acts, pairs)
    c = cos(v, true_dir)
    half = len(pairs) // 2
    va, _ = diff_of_means(acts, pairs[:half])
    vb, _ = diff_of_means(acts, pairs[half:])
    c_half = cos(va, vb)
    rand = rng.normal(size=(N_RANDOM, d))
    floor = max(abs(cos(v, r / np.linalg.norm(r))) for r in rand)

    print(f"planted-direction recovery: cos(v, true_dir) = {c:.4f}  (n={got_n} pairs, d={d})")
    print(f"split-half cosine          = {c_half:.4f}   random floor |cos| max = {floor:.4f}")
    assert got_n == n, "pair count wrong"
    assert c > 0.9, f"difference-of-means failed to recover the planted direction (cos={c:.3f})"
    assert c_half > 0.8, f"split-half cosine implausibly low ({c_half:.3f})"
    assert floor < 0.3, f"random floor too high ({floor:.3f})"
    assert v.shape == (d,), "vector must be [d], not [n]"
    print("SELF-TEST PASS: difference-of-means recovers the planted direction; the common "
          "offset cancels; split-half >> random floor")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-filter", action="store_true",
                    help="recompute on ALL raw pairs -> vectors_nofilter.npz (filter-bias check)")
    args = ap.parse_args()
    self_test() if args.self_test else extract(args.run_dir, use_filter=not args.no_filter)
