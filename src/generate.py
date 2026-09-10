"""Generation pass — rollouts only, never activations.
Engine: vLLM, matching persona_vectors eval/eval_persona.py L58-81; our sampling
constants are native vLLM params and carry over verbatim. System-prompt composition
mirrors eval/eval_persona.py L149-154. Run GPU-side:
  .venv-gen/bin/python -m src.generate --smoke  # 8 responses + throughput estimate
  .venv-gen/bin/python -m src.generate          # pilot preset, 3000 responses
  .venv-gen/bin/python -m src.generate --grid   # full grid, 24,000 responses
(generation runs in .venv-gen — the vllm venv; see requirements.txt header)"""

import argparse
import datetime
import json
import subprocess
import time
from collections import Counter

from src.config import (MODEL_NAME, SEED, TEMPERATURE, TOP_P, MAX_NEW_TOKENS,
                        MIN_NEW_TOKENS, ASSISTANT_NAME_POS, ASSISTANT_NAME_NEG,
                        EXTRACT_JSON, EVAL_JSON, PERSONAS_JSON, RUNS_DIR,
                        model_spec, TP_SIZE_32B, MAX_MODEL_LEN_32B)

SPLIT_JSON = {"extract": EXTRACT_JSON, "eval": EVAL_JSON}
# pilot preset: (persona_id, split); n_per_question=3 → 5 persona-splits × 2 pol × 5 instr × 20 q × 3 = 3000
PILOT_JOBS = [("default", "extract"), ("default", "eval"), ("pirate", "extract"),
              ("pirate", "eval"), ("consultant", "eval")]
# grid preset: 6 personas (pilot + grid tiers of data/personas/pilot.json), extract AND eval
# everywhere, n_per_question=10 → 6 × 2 splits × 2 pol × 5 instr × 20 q × 10 = 24,000
GRID_PERSONAS = ["default", "consultant", "pirate", "teacher", "therapist", "hermit"]
GRID_JOBS = [(p, s) for p in GRID_PERSONAS for s in ("extract", "eval")]
GRID_N_PER_QUESTION = 10
# Qwen3-32B replication preset: default+pirate only, both splits, n=5 (OURS, time-boxed)
# -> 2 personas x 2 splits x 2 pol x 5 instr x 20 q x 5 = 4,000 responses
Q32B_JOBS = [(p, s) for p in ("default", "pirate") for s in ("extract", "eval")]
Q32B_N_PER_QUESTION = 5
# measured on a 3000-response pilot: 77.5 s wall (7614.7 tok/s), one A100
REF_RESPONSES_PER_S = 3000 / 77.5


def chat_prompt(tok, system, question, thinking_off=False):
    """THE single prompt definition, used by generation AND the teacher-forced pass in
    src/activations.py. If these two ever diverge, prompt_len is wrong and every stored
    activation is silently misaligned. Qwen3 emits <think> blocks unless thinking is off."""
    kw = {"enable_thinking": False} if thinking_off else {}
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True, **kw)


def a_or_an(word):  # copied from external/persona_vectors/eval/eval_persona.py L134-135
    return "an" if word[0].lower() in "aeiou" else "a"


def system_prompt(noun, polarity, instruction):
    """noun=None → persona_vectors template UNCHANGED (eval/eval_persona.py L149-154);
    else our persona variant: persona noun + stay-in-character clause, trait
    instruction text byte-identical from the artifact."""
    name = ASSISTANT_NAME_POS if polarity == "pos" else ASSISTANT_NAME_NEG
    if noun is None:
        return f"You are {a_or_an(name)} {name} assistant. {instruction}"
    return (f"You are {a_or_an(name)} {name} {noun}; stay fully in character as "
            f"{a_or_an(noun)} {noun}. {instruction}")


def build_records(jobs, n_per_question, n_instructions=None, n_questions=None):
    nouns = {p["id"]: p["noun"] for p in json.load(open(PERSONAS_JSON))["personas"]}
    records = []
    for persona_id, split in jobs:
        data = json.load(open(SPLIT_JSON[split]))
        for polarity in ("pos", "neg"):
            for i, ins in enumerate(data["instruction"][:n_instructions]):
                for q, question in enumerate(data["questions"][:n_questions]):
                    for s in range(n_per_question):
                        records.append(dict(
                            response_id=f"{persona_id}|{polarity}|{split}|i{i}|q{q}|s{s}",
                            persona=persona_id, polarity=polarity, split=split,
                            instruction_idx=i, question_idx=q, sample_idx=s,
                            system_prompt=system_prompt(nouns[persona_id], polarity, ins[polarity]),
                            question=question))
    return records


def git_hash():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", default=None, help="persona ids (with --splits overrides the pilot preset)")
    ap.add_argument("--splits", nargs="+", default=None, choices=["extract", "eval"])
    ap.add_argument("--n-per-question", type=int, default=3)
    ap.add_argument("--smoke", action="store_true",
                    help="default+pirate, both polarities, 1 instr x 2 q x 1 sample, eval only")
    ap.add_argument("--model-32b", action="store_true",
                    help="Qwen3-32B replication preset: default+pirate, extract+eval, n=5")
    ap.add_argument("--grid", action="store_true",
                    help=f"full grid: {len(GRID_PERSONAS)} personas x extract+eval, "
                         f"n_per_question={GRID_N_PER_QUESTION}")
    args = ap.parse_args()

    spec = model_spec(args.model_32b)
    if args.smoke:
        jobs, records = [("default", "eval"), ("pirate", "eval")], None
        records = build_records(jobs, 1, n_instructions=1, n_questions=2)
    elif args.model_32b:
        records = build_records(Q32B_JOBS, Q32B_N_PER_QUESTION)
    elif args.grid:
        records = build_records(GRID_JOBS, GRID_N_PER_QUESTION)
    elif args.personas:
        jobs = [(p, s) for p in args.personas for s in (args.splits or ["extract", "eval"])]
        records = build_records(jobs, args.n_per_question)
    else:
        records = build_records(PILOT_JOBS, args.n_per_question)

    kind = "q32b" if args.model_32b else "grid" if args.grid else "pilot"
    run_id = f"{datetime.date.today()}-{kind}-s{SEED}" + ("-smoke" if args.smoke else "")
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists — refusing to overwrite; move or remove it first.")

    by_persona = Counter(r["persona"] for r in records)
    print(f"run_id {run_id}: {len(records)} responses over {len(by_persona)} personas x "
          f"{len({r['split'] for r in records})} splits x 2 polarities")
    print("  " + ", ".join(f"{k}:{v}" for k, v in sorted(by_persona.items())))
    print(f"  model {spec['name']}, thinking_off={spec['thinking_off']}")
    print(f"  ETA ~{len(records) / REF_RESPONSES_PER_S / 60:.1f} min at the 7B's measured "
          f"{REF_RESPONSES_PER_S:.1f} responses/s (the 32B will be SLOWER — this is a floor)")

    from transformers import AutoTokenizer  # deferred: --help must work without GPU libs
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(spec["name"])
    # mirrors eval/eval_persona.py L58-70 (SamplingParams incl. stop=[eos], chat template);
    # per-request seeds are OURS (they set none): SEED*1_000_000 + record index
    prompts = [chat_prompt(tok, r["system_prompt"], r["question"], spec["thinking_off"])
               for r in records]
    params = [SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS,
                             min_tokens=MIN_NEW_TOKENS, skip_special_tokens=True,
                             stop=[tok.eos_token], seed=SEED * 1_000_000 + i)
              for i in range(len(records))]
    extra = dict(tensor_parallel_size=TP_SIZE_32B, max_model_len=MAX_MODEL_LEN_32B) \
        if args.model_32b else {}
    llm = LLM(model=spec["name"], dtype="bfloat16", seed=SEED, **extra)

    t0 = time.time()
    outs = llm.generate(prompts, params)
    wall = time.time() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)

    run_dir.mkdir(parents=True)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    gen_params = dict(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS,
                      min_tokens=MIN_NEW_TOKENS, seed_base=SEED)
    with open(run_dir / "responses.jsonl", "w") as f:
        for i, (r, o) in enumerate(zip(records, outs)):
            f.write(json.dumps(dict(run_id=run_id, **r, response=o.outputs[0].text,
                                    gen_params=gen_params, seed=SEED * 1_000_000 + i,
                                    timestamp=stamp)) + "\n")
    (run_dir / "run_meta.json").write_text(json.dumps(dict(
        run_id=run_id, model=spec["name"], git=git_hash(), n_responses=len(records),
        thinking_off=spec["thinking_off"], layer=spec["layer"], d_model=spec["d_model"],
        gen_params=gen_params, wall_seconds=round(wall, 1),
        tok_per_s=round(n_tok / wall, 1), argv=vars(args)), indent=2))
    print(f"{run_id}: {len(records)} responses -> {run_dir}/responses.jsonl")

    if args.smoke:
        rps = len(records) / wall
        print(f"throughput: {n_tok / wall:.0f} tok/s, {wall:.1f}s wall for {len(records)} responses")
        print(f"extrapolated: full pilot (3000 resp) ~{3000 / rps / 60:.0f} min, "
              f"full grid (24000 resp) ~{24000 / rps / 60:.0f} min (batching will beat this)")


if __name__ == "__main__":
    main()
