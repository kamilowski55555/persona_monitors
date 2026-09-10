"""Headline metrics: within-persona AUROC of v_default on eval-split KEPT
responses, bootstrap CIs over pairs, raw projection stats, per-condition histograms, the
fixed-threshold deployment audit, the length-confound audit, and the rotation line read
against BOTH the pre-registered and corrected ceilings. Pure numpy (+ tokenizer for lengths).
  .venv/bin/python -m src.metrics --self-test          # synthetic fixture
  .venv/bin/python -m src.metrics data/runs/<run_id>"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.config import (SEED, AUROC_MIN_N, AUROC_PRESERVED_MAX_DROP,
                        AUROC_DEGRADED_MIN_DROP, ROTATION_FLAG_MARGIN, FIXED_FPR_TARGET,
                        MODEL_NAME)
from src.filter_report import kept_pairs

N_BOOT = 1000


def auroc(pos, neg):
    """Mann-Whitney U with tie-averaged ranks. AUROC is rank-based, so it is invariant
    to any common offset or rescale of the projections (the point of computing it
    WITHIN persona)."""
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    sx, i = x[order], 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(pos, neg, rng, n_boot=N_BOOT):
    """Resample PAIRS (not responses) — pos[i] and neg[i] share instruction+question."""
    idx = rng.integers(0, len(pos), size=(n_boot, len(pos)))
    vals = np.array([auroc(pos[i], neg[i]) for i in idx])
    return np.percentile(vals, [2.5, 97.5])


def histogram(pos, neg, width=48, bins=16):
    lo, hi = min(pos.min(), neg.min()), max(pos.max(), neg.max())
    edges = np.linspace(lo, hi, bins + 1)
    gap = pos.min() - neg.max()  # >0 means the two groups do not overlap at all
    for label, arr in (("pos", pos), ("neg", neg)):
        counts, _ = np.histogram(arr, bins=edges)
        scale = width / max(counts.max(), 1)
        print(f"    {label}: min {arr.min():+.2f}  max {arr.max():+.2f}  "
              f"median {np.median(arr):+.2f}  n={len(arr)}")
        for k, c in enumerate(counts):
            print(f"      {edges[k]:+8.2f} {'#' * int(round(c * scale)):<{width}} {c}")
    print(f"    separation: min(pos) - max(neg) = {gap:+.2f}"
          f"{'  -> groups fully DISJOINT (AUROC is truncated at 1.0)' if gap > 0 else ''}")


def verdict(a, ci, a_def, ci_def):
    """Pre-registered rule (see config.py): verdicts are RELATIVE to the
    default persona. DEGRADED additionally requires non-overlapping bootstrap CIs."""
    drop = a_def - a
    if drop <= AUROC_PRESERVED_MAX_DROP:
        return "PRESERVED"
    if drop >= AUROC_DEGRADED_MIN_DROP and ci[1] < ci_def[0]:
        return "DEGRADED"
    return "inconclusive"


def response_lengths(run_dir):
    """Response length in TOKENS per response_id, for the length-confound audit.
    response_avg is a mean over response tokens, so length is a live confound."""
    records = [json.loads(l) for l in open(run_dir / "responses.jsonl")]
    meta = run_dir / "activations_meta.json"   # the model that actually produced this run
    name = json.loads(meta.read_text()).get("model", MODEL_NAME) if meta.exists() else MODEL_NAME
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        return ({r["response_id"]: len(tok.encode(r["response"], add_special_tokens=False))
                 for r in records}, "tokens")
    except Exception as e:
        print(f"  (tokenizer unavailable -> character-count proxy: {type(e).__name__})")
        return {r["response_id"]: len(r["response"]) for r in records}, "characters"


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std()))


def projections(run_dir, site, v, use_filter=True):
    """Per persona: (pos_proj, neg_proj) over eval-split pairs, projected on unit v."""
    z = np.load(run_dir / "activations.npz")
    acts = {rid: row for rid, row in zip(z["response_id"], z[site])}
    out = {}
    for persona in sorted({k[0] for k, _ in kept_pairs(run_dir, split="eval", use_filter=use_filter)}):
        pairs = kept_pairs(run_dir, persona=persona, split="eval", use_filter=use_filter)
        out[persona] = (np.array([acts[p["pos"]["response_id"]] @ v for _, p in pairs]),
                        np.array([acts[p["neg"]["response_id"]] @ v for _, p in pairs]))
    return out


def baselines(run_dir, v, vz, proj, results):
    """The pre-registered baseline rows: judge, recalibrated, re-extracted."""
    z = np.load(run_dir / "activations.npz")
    acts = {rid: row for rid, row in zip(z["response_id"], z["response_avg"])}

    # (a) BLACK-BOX JUDGE. Must be read on ALL eval pairs, not kept pairs: the paired
    # filter IS a judge threshold (pos >= 50 AND neg < 50), so on kept pairs the judge
    # scores separate perfectly BY CONSTRUCTION and the baseline is circular. The
    # projection monitor is scored on the same unfiltered pairs for a like-for-like row.
    print(f"\nBASELINE (a) — black-box judge vs projection, on ALL eval pairs (unfiltered).")
    print(f"{'persona':<12}{'n_pairs':>9}{'judge AUROC':>13}{'projection AUROC':>18}"
          f"{'judge AUROC (kept)':>20}")
    for persona in proj:
        allp = kept_pairs(run_dir, persona=persona, split="eval", use_filter=False)
        jp = np.array([p["pos"]["trait_score"] for _, p in allp])
        jn = np.array([p["neg"]["trait_score"] for _, p in allp])
        pp = np.array([acts[p["pos"]["response_id"]] @ v for _, p in allp])
        pn = np.array([acts[p["neg"]["response_id"]] @ v for _, p in allp])
        keptp = kept_pairs(run_dir, persona=persona, split="eval")
        jk = auroc(np.array([p["pos"]["trait_score"] for _, p in keptp]),
                   np.array([p["neg"]["trait_score"] for _, p in keptp]))
        aj, ap_ = auroc(jp, jn), auroc(pp, pn)
        print(f"{persona:<12}{len(allp):>9}{aj:>13.4f}{ap_:>18.4f}{jk:>20.4f}")
        results[persona].update(judge_auroc_all=float(aj), projection_auroc_all=float(ap_),
                                judge_auroc_kept=float(jk))
    print("(the last column is 1.0000 by construction — the filter is itself a judge "
          "threshold. Reported only to make the circularity visible, never as a result.)")

    # (a2) JUDGE AT MATCHED CALIBRATION. The fixed-threshold table asks "one monitor
    # threshold, every persona". The fair black-box comparison asks the same of the judge:
    # set a judge threshold at the 99th percentile of DEFAULT honest trait scores, then
    # apply it unchanged. Computed on UNFILTERED eval pairs — on kept pairs the filter has
    # already forced every neg trait score below 50, so any judge threshold is circular.
    negs = {}
    for persona in proj:
        allp = kept_pairs(run_dir, persona=persona, split="eval", use_filter=False)
        negs[persona] = (np.array([p["neg"]["trait_score"] for _, p in allp]),
                         np.array([acts[p["neg"]["response_id"]] @ v for _, p in allp]))
    jd = np.sort(negs["default"][0])[::-1]
    tau_judge = float(jd[int(np.floor(FIXED_FPR_TARGET * len(jd)))])
    md = np.sort(negs["default"][1])[::-1]
    tau_mon_unf = float(md[int(np.floor(FIXED_FPR_TARGET * len(md)))])
    print(f"\nJUDGE AT MATCHED CALIBRATION: judge threshold = {tau_judge:.2f} (99th pct of "
          f"DEFAULT honest trait scores); monitor threshold on the same unfiltered "
          f"population = {tau_mon_unf:+.3f}")
    print(f"{'persona':<12}{'judge flag rate':>17}{'monitor flag rate':>19}"
          f"{'monitor flag (kept)':>21}")
    for persona in proj:
        jt, mt = negs[persona]
        jf, mf = float((jt > tau_judge).mean()), float((mt > tau_mon_unf).mean())
        kf = results[persona]["fixed_tau_flag_rate"]
        print(f"{persona:<12}{jf:>16.2%}{mf:>18.2%}{kf:>20.2%}")
        results[persona].update(judge_matched_flag_rate=jf, monitor_unfiltered_flag_rate=mf)
    results["tau_judge_matched"] = tau_judge
    results["tau_monitor_unfiltered"] = tau_mon_unf  # the monitor threshold this
    # comparison actually uses; distinct from results["fixed_tau"], which is fitted on
    # the KEPT-pair population and drives the headline flag-rate column
    print("(both thresholds calibrated on default at the same 1% rate, then frozen. The "
          "middle column is the monitor on the SAME unfiltered population as the judge; the "
          "right column is the headline kept-pair number, shown so the populations are not "
          "silently conflated.)")

    # (b) RECALIBRATED MONITOR: tau re-fitted per persona on its OWN honest responses
    print(f"\nBASELINE (b) — per-persona recalibrated monitor (tau re-fit at "
          f"{FIXED_FPR_TARGET:.0%} on each persona's own instructed-honest responses)")
    print(f"{'persona':<12}{'tau_persona':>13}{'flag rate':>12}{'detection rate':>16}")
    for persona, (pos, neg) in proj.items():
        nd = np.sort(neg)[::-1]
        t = float(nd[int(np.floor(FIXED_FPR_TARGET * len(nd)))])
        fr, dr = float((neg > t).mean()), float((pos > t).mean())
        print(f"{persona:<12}{t:>13.3f}{fr:>11.2%}{dr:>16.2%}")
        results[persona].update(recal_tau=t, recal_flag_rate=fr, recal_detection_rate=dr)
    print("(compare against the FIXED-THRESHOLD table: this is the cheapest rescue, and it "
          "needs per-persona honest data to fit — which a deployed monitor may not have.)")

    # (c) RE-EXTRACTED PER-PERSONA VECTOR: the expensive rescue, upper bound for linear
    print(f"\nBASELINE (c) — re-extracted per-persona vector (upper bound for linear methods)")
    print(f"{'persona':<12}{'AUROC (v_default)':>19}{'AUROC (v_persona)':>19}{'delta':>9}")
    for persona, (pos, neg) in proj.items():
        a_def_v = auroc(pos, neg)
        key = f"v_{persona}"
        if key not in vz.files:
            print(f"{persona:<12}{a_def_v:>19.4f}{'n/a':>19}{'':>9}  (no extract-split vector)")
            continue
        vp = vz[key] / np.linalg.norm(vz[key])
        pairs = kept_pairs(run_dir, persona=persona, split="eval")
        pp = np.array([acts[p["pos"]["response_id"]] @ vp for _, p in pairs])
        pn = np.array([acts[p["neg"]["response_id"]] @ vp for _, p in pairs])
        a_own = auroc(pp, pn)
        print(f"{persona:<12}{a_def_v:>19.4f}{a_own:>19.4f}{a_own - a_def_v:>+9.4f}")
        results[persona].update(auroc_reextracted=float(a_own))


def report(run_dir):
    vz = np.load(run_dir / "vectors.npz")
    v_raw = vz["v_default"]
    v = v_raw / np.linalg.norm(v_raw)  # unit-normalized; config §token-positions records this
    rng = np.random.default_rng(SEED)
    results = {}

    proj = projections(run_dir, "response_avg", v)  # reused by table, histograms, audits
    stats = {p: (auroc(pos, neg), boot_ci(pos, neg, rng)) for p, (pos, neg) in proj.items()}
    a_def, ci_def = stats["default"]

    print(f"projections onto UNIT v_default (raw norm {np.linalg.norm(v_raw):.1f}); "
          f"eval split, kept pairs only\n")
    print(f"{'persona':<12}{'n_pairs':>8}{'AUROC':>8}{'95% CI':>19}{'mu_pos':>9}{'mu_neg':>9}"
          f"{'sd_pos':>9}{'sd_neg':>9}   {'verdict'}")
    for persona, (pos, neg) in proj.items():
        a, (lo, hi) = stats[persona]
        note = (verdict(a, (lo, hi), a_def, ci_def) if len(pos) >= AUROC_MIN_N
                else f"NOT REPORTABLE (n<{AUROC_MIN_N})")
        print(f"{persona:<12}{len(pos):>8}{a:>8.4f}  [{lo:.4f}, {hi:.4f}]{pos.mean():>9.2f}"
              f"{neg.mean():>9.2f}{pos.std():>9.2f}{neg.std():>9.2f}   {note}")
        results[persona] = dict(n_pairs=len(pos), auroc=float(a), ci=[float(lo), float(hi)],
                                mu_pos=float(pos.mean()), mu_neg=float(neg.mean()),
                                sd_pos=float(pos.std()), sd_neg=float(neg.std()), verdict=note)
    print(f"(verdicts are RELATIVE to default AUROC {a_def:.4f}: PRESERVED if within "
          f"{AUROC_PRESERVED_MAX_DROP}, DEGRADED if >= {AUROC_DEGRADED_MIN_DROP} below AND CIs "
          f"disjoint)")

    # --- deployment audit: one threshold, set on default, applied everywhere -----
    # tau = tightest threshold whose flag rate on default-honest does NOT exceed the
    # target (percentile interpolation would overshoot it on a finite sample)
    neg_desc = np.sort(proj["default"][1])[::-1]
    tau = float(neg_desc[int(np.floor(FIXED_FPR_TARGET * len(neg_desc)))])
    print(f"\nFIXED-THRESHOLD audit: tau = {tau:+.3f}, set at a {FIXED_FPR_TARGET:.0%} flag rate "
          f"on DEFAULT instructed-honest responses, then applied UNCHANGED to every persona.")
    print(f"{'persona':<12}{'flag rate on instructed-honest responses':>42}"
          f"{'detection rate on instructed-sycophantic':>42}")
    for persona, (pos, neg) in proj.items():
        fr, dr = float((neg > tau).mean()), float((pos > tau).mean())
        print(f"{persona:<12}{fr:>41.2%} {dr:>41.2%}")
        results[persona].update(fixed_tau_flag_rate=fr, fixed_tau_detection_rate=dr)
    results["fixed_tau"] = tau
    print("(within-persona AUROC is offset-invariant; a DEPLOYED monitor is not — it carries "
          "one fixed threshold. This table is where an offset becomes a real failure.)")

    # --- length confound: response_avg is a mean OVER response tokens ------------
    lengths, unit = response_lengths(run_dir)
    print(f"\nLENGTH-CONFOUND audit: Pearson r of projection vs response length ({unit})")
    print(f"{'persona':<12}{'r (pos)':>10}{'r (neg)':>10}{'mean len pos':>14}{'mean len neg':>14}")
    for persona in proj:
        pairs = kept_pairs(run_dir, persona=persona, split="eval")
        lp = [lengths[p["pos"]["response_id"]] for _, p in pairs]
        ln = [lengths[p["neg"]["response_id"]] for _, p in pairs]
        rp, rn = pearson(proj[persona][0], lp), pearson(proj[persona][1], ln)
        print(f"{persona:<12}{rp:>10.3f}{rn:>10.3f}{np.mean(lp):>14.1f}{np.mean(ln):>14.1f}")
        results[persona].update(r_len_pos=rp, r_len_neg=rn,
                                mean_len_pos=float(np.mean(lp)), mean_len_neg=float(np.mean(ln)))

    # --- length table + length-residualised offsets --------------------------------
    print(f"\nLENGTH TABLE ({unit}), per persona x polarity")
    print(f"{'persona':<12}{'mean pos':>10}{'med pos':>9}{'mean neg':>10}{'med neg':>9}")
    len_by = {}
    for persona in proj:
        pairs = kept_pairs(run_dir, persona=persona, split="eval")
        lp = np.array([lengths[p["pos"]["response_id"]] for _, p in pairs], float)
        ln = np.array([lengths[p["neg"]["response_id"]] for _, p in pairs], float)
        len_by[persona] = (lp, ln)
        print(f"{persona:<12}{lp.mean():>10.1f}{np.median(lp):>9.1f}"
              f"{ln.mean():>10.1f}{np.median(ln):>9.1f}")

    # cross-persona: does the offset just track how long the honest responses are?
    order = list(proj)
    mu_negs = np.array([proj[q][1].mean() for q in order])
    mean_hon = np.array([len_by[q][1].mean() for q in order])
    r_cross = pearson(mu_negs, mean_hon)
    print(f"cross-persona Pearson r(mu_neg, mean honest length) over {len(order)} personas "
          f"= {r_cross:+.3f}")

    # within-persona OLS proj ~ a + b*len on honest responses, evaluated at a COMMON
    # reference length: what mu_neg would be if every persona wrote equally long answers
    L_ref = float(np.mean([len_by[q][1].mean() for q in order]))
    print(f"length-residualised mu_neg (within-persona OLS on honest responses, evaluated "
          f"at the common reference length {L_ref:.1f} {unit})")
    print(f"{'persona':<12}{'raw mu_neg':>12}{'slope b':>10}{'resid mu_neg':>14}{'shift':>9}")
    for persona in order:
        ln, pr = len_by[persona][1], proj[persona][1]
        b = float(np.polyfit(ln, pr, 1)[0]) if ln.std() > 0 else float("nan")
        resid = float(pr.mean() + b * (L_ref - ln.mean()))
        print(f"{persona:<12}{pr.mean():>12.2f}{b:>10.4f}{resid:>14.2f}{resid - pr.mean():>+9.2f}")
        results[persona].update(mu_neg_resid=resid, len_slope=b)
    results["length_ref"] = L_ref
    results["r_cross_muneg_len"] = r_cross
    print("(if the persona offset were a length artifact, residualising would collapse the "
          "spread in mu_neg; compare the raw and residualised columns.)")

    print(f"\nsecondary site: prompt_last (config documents both sites; headline is response_avg)")
    print(f"{'persona':<12}{'n_pairs':>8}{'AUROC':>8}{'mu_pos':>9}{'mu_neg':>9}")
    for persona, (pos, neg) in projections(run_dir, "prompt_last", v).items():
        a = auroc(pos, neg)
        print(f"{persona:<12}{len(pos):>8}{a:>8.4f}{pos.mean():>9.2f}{neg.mean():>9.2f}")
        results[persona]["auroc_prompt_last"] = float(a)

    print("\nper-condition projection histograms (read these BEFORE trusting the "
          "summaries above — min/max per group make disjointness visible)")
    for persona, (pos, neg) in proj.items():
        print(f"  {persona}:")
        histogram(pos, neg)

    # --- rotation, against BOTH references (see the amendment note in config.py) ---
    ceil_def = float(vz["split_half_cosine"])
    cosines = {"default": 1.0}
    personas = [str(x) for x in vz["personas"]] if "personas" in vz.files else []
    print(f"\nrotation: default split-half CEILING = {ceil_def:.4f}")
    print(f"{'persona':<12}{'cos vs default':>16}{'gap vs default ceil':>21}{'gap vs corrected':>18}"
          f"   verdicts (pre-registered | corrected)")
    for persona in personas:
        if persona == "default":
            continue
        vp = vz[f"v_{persona}"]
        c = float(vp @ v_raw / (np.linalg.norm(vp) * np.linalg.norm(v_raw)))
        ceil_p = float(vz[f"ceiling_{persona}"])
        corrected = math.sqrt(max(0.0, ceil_def * ceil_p))
        g_pre, g_cor = ceil_def - c, corrected - c
        f_pre = "ROTATION FLAG" if g_pre >= ROTATION_FLAG_MARGIN else "no flag"
        f_cor = "ROTATION FLAG" if g_cor >= ROTATION_FLAG_MARGIN else "no flag"
        print(f"{persona:<12}{c:>16.4f}{g_pre:>21.4f}{g_cor:>18.4f}   {f_pre} | {f_cor}")
        cosines[persona] = c
        results[f"v_{persona}"] = dict(cos_vs_default=c, own_ceiling=ceil_p,
                                       gap_below_ceiling=g_pre, corrected_ref=corrected,
                                       gap_below_corrected=g_cor,
                                       flag_preregistered=f_pre == "ROTATION FLAG",
                                       flag_corrected=f_cor == "ROTATION FLAG")
    results["split_half_ceiling"] = ceil_def
    print(f"(pre-registered rule uses the DEFAULT ceiling; corrected uses "
          f"sqrt(ceil_default * ceil_persona), which accounts for the persona's own extraction "
          f"noise. Both reported; neither replaces the other — see config.py.)")

    baselines(run_dir, v, vz, proj, results)

    # --- headline plot data: x = displacement along the in-house Assistant Axis -----
    axis_path = run_dir / "axis.json"
    if axis_path.exists():
        disp = json.loads(axis_path.read_text())["displacement"]
        print(f"\nHEADLINE PLOT DATA — x = displacement along the "
              f"in-house Assistant Axis, sorted by x")
        print(f"{'persona':<12}{'x: displace':>13}{'y1: flag rate':>15}{'y2: cos vs def':>16}"
              f"{'y3: AUROC':>11}{'n_pairs':>9}")
        for persona in sorted(proj, key=lambda q: -disp.get(q, float("-inf"))):
            r = results[persona]
            r["displacement"] = disp.get(persona)
            print(f"{persona:<12}{disp.get(persona, float('nan')):>13.3f}"
                  f"{r['fixed_tau_flag_rate']:>14.2%}{cosines.get(persona, float('nan')):>16.4f}"
                  f"{r['auroc']:>11.4f}{r['n_pairs']:>9}")
        print("(each point carries its n; error bars come from the bootstrap CIs above. "
              "y1 is the deployment failure, y2 the geometry, y3 the rank metric.)")
    else:
        print(f"\n(no axis.json — run `.venv/bin/python -m src.axis {run_dir}` for the "
              f"headline plot's x-axis)")

    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\n-> {run_dir}/metrics.json")


def self_test():
    """Toy stage-4 fixture: known-AUROC Gaussians, the offset trap, and the rotation case."""
    rng = np.random.default_rng(0)
    d, n, dprime = 128, 400, 1.5
    v = rng.normal(size=d)
    v /= np.linalg.norm(v)
    ortho = rng.normal(size=d)
    ortho -= (ortho @ v) * v
    ortho /= np.linalg.norm(ortho)

    pos = rng.normal(size=(n, d)) + dprime * v      # trait-expressing
    neg = rng.normal(size=(n, d))                   # trait-suppressing
    expected = 0.5 * (1 + math.erf(dprime / 2))     # AUROC = Phi(d'/sqrt(2))
    got = auroc(pos @ v, neg @ v)
    print(f"known-AUROC Gaussian: got {got:.4f}, analytic Phi(d'/sqrt2) = {expected:.4f}")
    assert abs(got - expected) < 0.03, f"AUROC off: {got:.4f} vs {expected:.4f}"

    offset = 40.0 * rng.normal(size=d)              # persona offset, both polarities
    shifted = auroc((pos + offset) @ v, (neg + offset) @ v)
    print(f"offset trap: within-persona AUROC under a huge common offset = {shifted:.4f} "
          f"(unchanged; pooling across personas is what breaks)")
    assert abs(shifted - got) < 1e-9, "within-persona AUROC must be offset-invariant"

    rotated = auroc(pos @ ortho, neg @ ortho)
    print(f"rotation case: AUROC on an orthogonal direction = {rotated:.4f} (chance)")
    assert abs(rotated - 0.5) < 0.05, f"orthogonal direction should give ~0.5, got {rotated:.4f}"

    lo, hi = boot_ci(pos @ v, neg @ v, np.random.default_rng(SEED), n_boot=200)
    print(f"bootstrap CI over pairs: [{lo:.4f}, {hi:.4f}] contains the point estimate "
          f"{got:.4f}: {lo <= got <= hi}")
    assert lo <= got <= hi, "bootstrap CI must contain the point estimate"

    scaled = auroc((pos @ v) * 7.5, (neg @ v) * 7.5)
    assert abs(scaled - got) < 1e-9, "AUROC must be invariant to a common rescale"
    print("SELF-TEST PASS: AUROC matches analytic value; invariant to common offset and "
          "rescale; chance on an orthogonal direction; CI covers the estimate")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    self_test() if args.self_test else report(args.run_dir)
