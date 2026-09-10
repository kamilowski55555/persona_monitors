"""Causal validation by steering. THE PROJECT'S ONE WRITE-PATH — the single
deliberate exception to "never hook during generation": a forward hook on model.model.layers[
STEER_HOOK_MODULE_IDX] adds alpha * v_hat to the block output DURING generation.
Raw HF + hooks in .venv, never vLLM. Shares no code path with collection.
  .venv/bin/python -m src.steering --self-test                    # CPU: hook logic + norm audit
  .venv/bin/python -m src.steering --self-test-gpu <run_dir>      # GPU: alpha=0 identity + shapes
  .venv/bin/python -m src.steering <run_dir>                      # the sweep -> responses.jsonl
  .venv/bin/python -m src.judge   data/runs/<sweep_id>            # existing judge, cache applies
  .venv/bin/python -m src.steering <run_dir> --report             # per-alpha table
--persona <id> composes the persona system prompt WITHOUT any trait instruction (Tier 2)."""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch

from src.config import (MODEL_NAME, SEED, LAYER, STEER_HOOK_MODULE_IDX, TEMPERATURE, TOP_P,
                        MAX_NEW_TOKENS, STEER_COEFS, STEER_POSITIONS, STEER_N_PER_QUESTION,
                        STEER_VERDICT_COEF, STEER_ABS_ALPHAS, STEER_HEADROOM_PRESERVED_MIN,
                        STEER_HEADROOM_DEGRADED_MAX, STEER_PIRATE_BASELINE_BAND,
                        EVAL_JSON, PERSONAS_JSON, RUNS_DIR)
from src.generate import a_or_an, git_hash

N_BOOT_HEADROOM = 1000


class Steerer:
    """Adds alpha * v_hat to the layer's output. positions="response" replicates
    persona_vectors activation_steer.py L88-91 (t[:, -1, :] += steer), which covers the
    prefill shape [1, seq, D] and every decode step [1, 1, D] with the same expression."""

    def __init__(self, model, v_hat, alpha, positions=STEER_POSITIONS):
        self.model, self.alpha, self.positions = model, float(alpha), positions
        p = next(model.parameters())
        self.steer = (self.alpha * torch.as_tensor(v_hat)).to(dtype=p.dtype, device=p.device)
        self.shapes, self._handle = [], None

    def _add(self, t):
        self.shapes.append(tuple(t.shape))
        if self.alpha == 0.0:
            return t  # exact no-op: alpha=0 must reproduce the unsteered run bit-for-bit
        t2 = t.clone()
        if self.positions == "response":
            t2[:, -1, :] += self.steer.to(t.device)
        elif self.positions == "all":
            t2 += self.steer.to(t.device)
        else:
            raise ValueError(f"unsupported positions: {self.positions}")
        return t2

    def _hook(self, module, ins, out):
        # transformers 4.57.6 returns a plain tensor here; older versions a tuple
        # (verified by hook_diagnostic.py) — handle both, like theirs L96-105
        if torch.is_tensor(out):
            return self._add(out)
        if isinstance(out, (tuple, list)) and torch.is_tensor(out[0]):
            return (self._add(out[0]), *out[1:])
        return out

    def __enter__(self):
        self._handle = self.model.model.layers[STEER_HOOK_MODULE_IDX].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle:
            self._handle.remove()
            self._handle = None


def steering_prompts(tok, persona=None):
    """Their steering eval uses NO explicit system message (eval_persona.py L108-113 with
    version=eval and no persona_instruction_type), so Qwen's chat template supplies its own
    default — and crucially NO trait instruction. --persona adds only the character frame."""
    questions = json.load(open(EVAL_JSON))["questions"]
    system = None
    if persona:
        noun = {p["id"]: p["noun"] for p in json.load(open(PERSONAS_JSON))["personas"]}[persona]
        if noun:  # OURS: our persona template with the trait clause stripped out (Tier 2)
            system = f"You are {a_or_an(noun)} {noun}; stay fully in character as {a_or_an(noun)} {noun}."
    out = []
    for q, question in enumerate(questions):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": question}]
        out.append((q, question, tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)))
    return out


def load_v_hat(run_dir, key="v_default"):
    v_raw = np.load(run_dir / "vectors.npz")[key]
    norm = float(np.linalg.norm(v_raw))
    return v_raw / norm, norm


def residual_norm(run_dir):
    """Typical single-token residual norm at LAYER, from the collected activations
    (prompt_last is one real token; response_avg is an average and reads smaller).
    Returns None when activations.npz is absent — a steering sweep only needs
    vectors.npz, so a deployment carrying just the vectors must still run. The audit
    then prints the absolute alphas without the residual-fraction column."""
    path = run_dir / "activations.npz"
    if not path.exists():
        return None
    return float(np.linalg.norm(np.load(path)["prompt_last"], axis=1).mean())


def norm_audit(run_dir, key="v_default"):
    v_hat, v_norm = load_v_hat(run_dir, key)
    ref = residual_norm(run_dir)
    where = (f"typical residual norm at layer {LAYER} = {ref:.1f} (mean ||prompt_last||)"
             if ref else f"residual norm UNAVAILABLE (no activations.npz in {run_dir.name})")
    print(f"NORM AUDIT — ||{key}_raw|| = {v_norm:.2f}, {where}")
    print(f"{'their coef':>11}{'alpha (unit)':>14}{'||alpha*v_hat||':>17}{'/ residual':>12}")
    for c in STEER_COEFS:
        a = c * v_norm
        frac = f"{a / ref:>11.1%}" if ref else f"{'n/a':>11}"
        print(f"{c:>11.2f}{a:>14.2f}{a:>17.2f}{frac}")
    print("(their coef multiplies the RAW vector — activation_steer.py L76 over "
          "eval_persona.py L259 — so alpha_unit = coef * ||v_raw||; see config UNIT WARNING)")
    return v_hat, v_norm, ref


def sweep(run_dir, persona=None, vector_key="v_default", abs_alpha=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    v_hat, v_norm, _ = norm_audit(run_dir, vector_key)
    # coef ladder (their convention) or matched-ABSOLUTE alphas (see config NORM TRAP)
    ladder = [(a / v_norm, a) for a in STEER_ABS_ALPHAS] if abs_alpha else \
             [(c, c * v_norm) for c in STEER_COEFS]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    prompts = steering_prompts(tok, persona)
    tag = f"-{vector_key}" if vector_key != "v_default" else ""
    tag += "-absA" if abs_alpha else ""
    sweep_id = f"{datetime.date.today()}-steer-{persona or 'default'}{tag}-s{SEED}"
    out_dir = RUNS_DIR / sweep_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists — refusing to overwrite.")
    out_dir.mkdir(parents=True)

    torch.manual_seed(SEED)
    # This loop stays unbatched by choice. Batching would save ~20 min of wall
    # clock and risks changing sampling behaviour in working code — declined, do not re-propose.
    rows = []
    for coef, alpha in ladder:
        for q, question, prompt in prompts:
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            with Steerer(model, v_hat, alpha) as st, torch.no_grad():
                out = model.generate(**enc, do_sample=TEMPERATURE > 0, temperature=TEMPERATURE,
                                     top_p=TOP_P, max_new_tokens=MAX_NEW_TOKENS,
                                     num_return_sequences=STEER_N_PER_QUESTION,
                                     pad_token_id=tok.eos_token_id)
            for s in range(STEER_N_PER_QUESTION):
                text = tok.decode(out[s][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                rows.append(dict(response_id=f"steer|{persona or 'default'}|c{coef}|q{q}|s{s}",
                                 persona=persona or "default", polarity="steer", split="steer",
                                 coef=coef, alpha=alpha, question_idx=q, sample_idx=s,
                                 question=question, response=text))
        print(f"  coef {coef:.2f} (alpha {alpha:.1f}) done — {len(rows)} responses so far", flush=True)

    with open(out_dir / "responses.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out_dir / "run_meta.json").write_text(json.dumps(dict(
        run_id=sweep_id, model=MODEL_NAME, git=git_hash(), persona=persona or "default",
        layer=LAYER, hook_module_idx=STEER_HOOK_MODULE_IDX, positions=STEER_POSITIONS,
        vector_key=vector_key, abs_alpha_mode=abs_alpha,
        coefs=[c for c, _ in ladder], v_raw_norm=v_norm, alphas=[a for _, a in ladder],
        n_per_question=STEER_N_PER_QUESTION, temperature=TEMPERATURE, seed=SEED,
        source_vectors=str(run_dir / "vectors.npz"), n_responses=len(rows)), indent=2))
    print(f"{len(rows)} responses -> {out_dir}/responses.jsonl\nNext: judge it, then --report")


def report(sweep_dir):
    scores = {s["response_id"]: s for s in map(json.loads, open(sweep_dir / "scores.jsonl"))}
    by_coef = {}
    for r in map(json.loads, open(sweep_dir / "responses.jsonl")):
        s = scores[r["response_id"]]
        by_coef.setdefault(r["coef"], {"trait": [], "coh": [], "refusals": 0})
        d = by_coef[r["coef"]]
        if s["trait_score"] is None or s["coherence_score"] is None:
            d["refusals"] += 1
            continue
        d["trait"].append(s["trait_score"])
        d["coh"].append(s["coherence_score"])

    coefs = sorted(by_coef)
    base = float(np.mean(by_coef[coefs[0]]["trait"]))  # trait at the lowest coef (0.0)
    print(f"{'coef':>7}{'n':>6}{'trait_mean':>12}{'trait_med':>11}{'coherence_mean':>16}"
          f"{'headroom':>10}{'refusals':>10}")
    traits, cohs, headrooms = [], [], {}
    for c in coefs:
        d = by_coef[c]
        t, h = float(np.mean(d["trait"])), float(np.mean(d["coh"]))
        hr = (t - base) / (100.0 - base)  # pre-registered 2026-09-03 (config)
        traits.append(t); cohs.append(h); headrooms[c] = hr
        print(f"{c:>7.2f}{len(d['trait']):>6}{t:>12.2f}{float(np.median(d['trait'])):>11.2f}"
              f"{h:>16.2f}{hr:>10.3f}{d['refusals']:>10}")
    print(f"(headroom claimed = (trait(coef) - trait(0)) / (100 - trait(0)); trait(0) = {base:.2f})")
    mono = all(b >= a for a, b in zip(traits, traits[1:]))
    print(f"\nmonotonicity of trait score in coef: {'YES' if mono else 'NO'} "
          f"({' -> '.join(f'{t:.1f}' for t in traits)})")
    if not mono:
        breaks = [f"{coefs[i]:.2f}->{coefs[i+1]:.2f}" for i in range(len(traits) - 1)
                  if traits[i + 1] < traits[i]]
        print(f"  non-monotone steps: {', '.join(breaks)}")
    cliff = [f"{c:.2f} (coh {h:.1f})" for c, h in zip(coefs, cohs) if h < 50]
    print(f"coherence cliff (mean coherence < 50): {', '.join(cliff) if cliff else 'none in sweep'}")
    print("(success = trait monotone in alpha UP TO the coherence breakdown; "
          "alphas past the cliff are not evidence of steering, only of damage)")

    # bootstrap the headroom CI by resampling QUESTIONS (the unit of independent
    # sampling here — 3 samples share a question), paired: the same resampled question
    # set is used for coef 0 and for the verdict coef, so the ratio stays coherent
    by_q = {}
    for r in map(json.loads, open(sweep_dir / "responses.jsonl")):
        sc = scores[r["response_id"]]
        if sc["trait_score"] is not None:
            by_q.setdefault(r["coef"], {}).setdefault(r["question_idx"], []).append(sc["trait_score"])
    if STEER_VERDICT_COEF in by_q and coefs[0] in by_q:
        qs = sorted(set(by_q[STEER_VERDICT_COEF]) & set(by_q[coefs[0]]))
        rng = np.random.default_rng(SEED)
        boots = []
        for _ in range(N_BOOT_HEADROOM):
            pick = rng.choice(qs, size=len(qs), replace=True)
            b = np.mean([v for q in pick for v in by_q[coefs[0]][q]])
            t = np.mean([v for q in pick for v in by_q[STEER_VERDICT_COEF][q]])
            boots.append((t - b) / (100.0 - b))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"headroom at coef {STEER_VERDICT_COEF}: {headrooms[STEER_VERDICT_COEF]:.3f} "
              f"95% CI [{lo:.3f}, {hi:.3f}]  ({N_BOOT_HEADROOM} bootstrap resamples over "
              f"{len(qs)} questions)")

    meta_path = sweep_dir / "run_meta.json"
    persona = json.loads(meta_path.read_text())["persona"] if meta_path.exists() else "?"
    if STEER_VERDICT_COEF in headrooms:
        hr = headrooms[STEER_VERDICT_COEF]
        v = ("PRESERVED" if hr >= STEER_HEADROOM_PRESERVED_MIN else
             "DEGRADED" if hr <= STEER_HEADROOM_DEGRADED_MAX else "inconclusive")
        coh_at = cohs[coefs.index(STEER_VERDICT_COEF)]
        print(f"\nSTEERING VERDICT ({persona}) at coef {STEER_VERDICT_COEF}: headroom "
              f"{hr:.3f} -> {v}  (PRESERVED >= {STEER_HEADROOM_PRESERVED_MIN}, DEGRADED <= "
              f"{STEER_HEADROOM_DEGRADED_MAX}; pre-registered 2026-09-03 pre-run)")
        if coh_at < 50:
            print(f"  CAVEAT: coherence at this coef is {coh_at:.1f} (< 50) — the verdict is "
                  f"read past the cliff and is not trustworthy")
    lo, hi = STEER_PIRATE_BASELINE_BAND
    if persona == "pirate":
        inb = lo <= base <= hi
        print(f"PRE-REGISTERED PREDICTION (pirate trait at coef 0 in [{lo:.0f}, {hi:.0f}]): "
              f"observed {base:.2f} -> {'CONFIRMED' if inb else 'MISSED'}")
    json.dump({str(c): dict(trait_mean=t, coherence_mean=h, headroom=headrooms[c],
                            n=len(by_coef[c]["trait"]), refusals=by_coef[c]["refusals"])
               for c, t, h in zip(coefs, traits, cohs)},
              open(sweep_dir / "steering_report.json", "w"), indent=2)
    print(f"\n-> {sweep_dir}/steering_report.json")


def self_test(run_dir=None):
    """CPU-only: the hook's shape handling on synthetic tensors, plus the norm audit."""
    d = 3584
    v_hat = np.zeros(d, dtype=np.float32); v_hat[0] = 1.0

    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))  # so .parameters() yields dtype/device
        def forward(self, x):
            return x
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([FakeLayer() for _ in range(LAYER)])
    m = FakeModel()

    alpha = 5.0
    st = Steerer(m, v_hat, alpha)
    for name, shape in (("prefill", (1, 7, d)), ("decode step", (1, 1, d))):
        t = torch.zeros(shape)
        got = st._add(t)
        assert got.shape == t.shape, f"{name}: shape changed {t.shape} -> {got.shape}"
        assert abs(got[0, -1, 0].item() - alpha) < 1e-6, f"{name}: last position not steered"
        assert got[0, -1, 1].item() == 0.0, f"{name}: steered off-direction"
        if shape[1] > 1:
            assert got[0, :-1, :].abs().max().item() == 0.0, "prefill: non-last positions touched"
        print(f"  {name} {shape}: last position += {alpha} on v_hat only, other positions untouched")
    assert st.shapes == [(1, 7, d), (1, 1, d)], "both shapes must be exercised"

    z = Steerer(m, v_hat, 0.0)
    t = torch.randn(1, 5, d)
    assert torch.equal(z._add(t), t), "alpha=0 must be an exact no-op"
    print("  alpha=0: hook returns the input tensor unchanged (exact no-op)")

    tup = st._hook(None, None, (torch.zeros(1, 3, d), "kvcache"))
    assert isinstance(tup, tuple) and tup[1] == "kvcache", "tuple outputs must pass extras through"
    plain = st._hook(None, None, torch.zeros(1, 3, d))
    assert torch.is_tensor(plain), "plain-tensor outputs must stay plain tensors"
    print("  hook handles BOTH a plain tensor (transformers 4.57.6) and a tuple, extras preserved")
    print("SELF-TEST PASS (CPU): hook shapes, alpha=0 no-op, tensor/tuple handling")
    if run_dir:
        print()
        norm_audit(run_dir)


def self_test_gpu(run_dir, persona=None):
    """GPU: alpha=0 with the hook ATTACHED must be token-identical to the hook DETACHED,
    at temperature 0, on 2 prompts; and a real 3-token generation must exercise both shapes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    v_hat, _ = load_v_hat(run_dir)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    prompts = [p for _, _, p in steering_prompts(tok, persona)[:2]]

    for i, prompt in enumerate(prompts):
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            plain = model.generate(**enc, do_sample=False, max_new_tokens=24,
                                   pad_token_id=tok.eos_token_id)
            with Steerer(model, v_hat, 0.0):
                hooked = model.generate(**enc, do_sample=False, max_new_tokens=24,
                                        pad_token_id=tok.eos_token_id)
        same = torch.equal(plain, hooked)
        print(f"  prompt {i}: alpha=0 hooked == detached, token-identical: {same}")
        assert same, f"prompt {i}: alpha=0 changed the output — the hook is not a no-op"

    enc = tok(prompts[0], return_tensors="pt", add_special_tokens=False).to(model.device)
    st = Steerer(model, v_hat, 1.0)
    with st, torch.no_grad():
        model.generate(**enc, do_sample=False, max_new_tokens=3, pad_token_id=tok.eos_token_id)
    prefill = [s for s in st.shapes if s[1] > 1]
    decode = [s for s in st.shapes if s[1] == 1]
    print(f"  3-token generation: {len(st.shapes)} hook calls — prefill {prefill[0]}, "
          f"{len(decode)} decode steps of shape {decode[0] if decode else None}")
    assert prefill and prefill[0][1] > 1, "no prefill-shaped call seen"
    assert len(decode) >= 2, f"expected >=2 decode-shaped calls, saw {len(decode)}"
    assert all(s[2] == 3584 for s in st.shapes), "hidden dim must be 3584 throughout"
    print("SELF-TEST PASS (GPU): alpha=0 is token-identical; both hook shapes exercised")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", type=Path, help="run dir holding vectors.npz")
    ap.add_argument("--persona", default=None, help="Tier 2: steer inside this persona")
    ap.add_argument("--vector-key", default="v_default",
                    help="key in vectors.npz to steer with (e.g. v_pirate)")
    ap.add_argument("--abs-alpha", action="store_true",
                    help="matched ABSOLUTE alphas on the unit vector (see config NORM TRAP) "
                         "— required for any cross-vector potency comparison")
    ap.add_argument("--report", action="store_true", help="aggregate a judged sweep")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--self-test-gpu", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(a.run_dir)
    elif a.self_test_gpu:
        self_test_gpu(a.run_dir, a.persona)
    elif a.report:
        report(a.run_dir)
    else:
        sweep(a.run_dir, a.persona, a.vector_key, a.abs_alpha)
