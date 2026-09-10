"""Teacher-forced activation collection — a SEPARATE pass over stored
rollouts, never hooked during generation. Raw HF, bf16 forward, fp32 before any stat.
Collects for EVERY response — filtering happens downstream, so metrics stay recomputable
under any filter without re-running the GPU.
  .venv/bin/python -m src.activations data/runs/<run_id> --limit 8   # self-test
  .venv/bin/python -m src.activations data/runs/<run_id>             # all responses"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import model_spec
from src.generate import chat_prompt


def collect(run_dir, limit=None, use_32b=False):
    spec = model_spec(use_32b)
    LAYER, D_MODEL = spec["layer"], spec["d_model"]
    records = [json.loads(l) for l in open(run_dir / "responses.jsonl")][:limit]
    tok = AutoTokenizer.from_pretrained(spec["name"])
    model = AutoModelForCausalLM.from_pretrained(spec["name"], torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    assert model.config.hidden_size == D_MODEL, \
        f"config says d_model={D_MODEL} but {spec['name']} has {model.config.hidden_size}"
    assert model.config.num_hidden_layers == spec["n_layers"], \
        f"config says {spec['n_layers']} layers but model has {model.config.num_hidden_layers}"

    resp_avg = np.zeros((len(records), D_MODEL), dtype=np.float32)
    prompt_last = np.zeros((len(records), D_MODEL), dtype=np.float32)
    ids, empty, boundary = [], [], 0
    t0 = time.time()
    for n, r in enumerate(records):
        # prompt built exactly as src/generate.py did; prompt_len convention from
        # cal_projection.py L83-84 (encode prompt alone, tokenize prompt+answer together)
        # IDENTICAL template to generation — shared definition, not a copy
        prompt = chat_prompt(tok, r["system_prompt"], r["question"], spec["thinking_off"])
        inputs = tok(prompt + r["response"], return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tok.encode(prompt, add_special_tokens=False))
        seq = inputs["input_ids"].shape[1]
        if seq <= prompt_len:  # empty/whitespace response: no response tokens to average
            empty.append(r["response_id"])
            ids.append(r["response_id"])
            continue
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states[LAYER][0].float()          # bf16 -> fp32 BEFORE any mean
        resp_avg[n] = hs[prompt_len:].mean(dim=0).cpu().numpy()   # mean over seq
        prompt_last[n] = hs[prompt_len - 1].cpu().numpy()
        ids.append(r["response_id"])
        # audit only: does the response span decode back to the stored response text?
        # (tokenizer boundary effects can shift one token; we keep their convention and count)
        span = tok.decode(inputs["input_ids"][0][prompt_len:])
        if span.strip()[:24] != r["response"].strip()[:24]:
            boundary += 1
        if n % 100 == 0 and n:
            rate = (n + 1) / (time.time() - t0)
            print(f"  {n + 1}/{len(records)}  {rate:.1f}/s  ETA {(len(records) - n - 1) / rate / 60:.1f} min",
                  flush=True)

    # a --limit run must never overwrite a full collection
    suffix = f"_limit{limit}" if limit else ""
    out_path = run_dir / f"activations{suffix}.npz"
    np.savez(out_path, response_id=np.array(ids), response_avg=resp_avg, prompt_last=prompt_last)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    (run_dir / f"activations_meta{suffix}.json").write_text(json.dumps(dict(
        layer=LAYER, model=spec["name"], d_model=D_MODEL,
        thinking_off=spec["thinking_off"], n=len(records), git=git,
        dtype="forward bf16, stored fp32 (cast before the mean)",
        site="response_avg = hidden_states[LAYER][prompt_len:].mean(0); prompt_last = [..., prompt_len-1]",
        empty_responses=empty, decode_boundary_mismatches=boundary,
        wall_seconds=round(time.time() - t0, 1)), indent=2))
    print(f"{len(records)} responses -> {out_path} ({time.time() - t0:.0f}s)")
    print(f"empty responses (all-zero rows, excluded downstream): {len(empty)}")
    print(f"response-span decode mismatches (audit, boundary tokens): {boundary}")
    return ids, resp_avg, prompt_last, empty, D_MODEL


def self_test(ids, resp_avg, prompt_last, empty, D_MODEL):
    n = len(ids)
    assert resp_avg.shape == (n, D_MODEL), f"response_avg {resp_avg.shape} != ({n}, {D_MODEL})"
    assert prompt_last.shape == (n, D_MODEL), f"prompt_last {prompt_last.shape} != ({n}, {D_MODEL})"
    assert resp_avg.dtype == np.float32 and prompt_last.dtype == np.float32, "must be fp32"
    assert len(set(ids)) == n, "response_ids must be unique"
    live = [i for i, rid in enumerate(ids) if rid not in set(empty)]
    for name, arr in (("response_avg", resp_avg), ("prompt_last", prompt_last)):
        assert not np.isnan(arr).any(), f"{name} contains NaN"
        assert not np.isinf(arr).any(), f"{name} contains inf"
        zero = [i for i in live if not arr[i].any()]
        assert not zero, f"{name} has all-zero rows at {zero[:5]}"
    print(f"SELF-TEST PASS: shapes ({n}, {D_MODEL}) fp32, no NaN/inf, no all-zero rows, ids unique")
    print(f"  |response_avg| mean norm {np.linalg.norm(resp_avg[live], axis=1).mean():.1f}, "
          f"|prompt_last| mean norm {np.linalg.norm(prompt_last[live], axis=1).mean():.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="collect only the first N (self-test)")
    ap.add_argument("--model-32b", action="store_true", help="Qwen3-32B (device_map auto, 3 GPUs)")
    args = ap.parse_args()
    self_test(*collect(args.run_dir, args.limit, args.model_32b))
