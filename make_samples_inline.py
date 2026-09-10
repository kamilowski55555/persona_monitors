"""Generate samples_inline.md — compact, uncurated response excerpts, categories fixed
and the draw within each seeded at 0. Full untruncated text is browsable in viewer.html.
Categories are fixed; the draw WITHIN each category is random at seed 0 and uncurated.
  .venv/bin/python make_samples_inline.py"""

import json
import random
import re
from pathlib import Path

SEED = 0
WORD_TARGET = 100
GRID = Path("data/runs/2026-09-03-grid-s0")
HOME_SWEEP = Path("data/runs/2026-09-03-steer-default-s0")
OUT = Path("samples_inline.md")


def load(run_dir):
    scores = {s["response_id"]: s for s in map(json.loads, open(run_dir / "scores.jsonl"))}
    recs = [json.loads(l) for l in open(run_dir / "responses.jsonl")]
    return scores, {r["response_id"]: r for r in recs}, recs


def one_line(t):
    return re.sub(r"\s+", " ", t).strip()


def trim(text):
    """Cut at the sentence boundary nearest WORD_TARGET words. The opening is never
    trimmed: if the first sentence already exceeds the target it is kept whole."""
    text = one_line(text)
    if len(text.split()) <= WORD_TARGET:
        return text
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", text)]
    if not ends:
        return text
    best = min(ends, key=lambda e: abs(len(text[:e].split()) - WORD_TARGET))
    return text[:best].rstrip() + " […truncated]"


def block(persona, condition, sc, question, response):
    return (f"**{persona} · {condition} · judge: trait {sc['trait_score']:.1f}, "
            f"coherence {sc['coherence_score']:.1f}**\n\n"
            f"*Q: {one_line(question)}*\n\n"
            f"> {trim(response)}\n")


rng = random.Random(SEED)
gsc, gidx, grecs = load(GRID)
out = ["# Inline response samples",
       "",
       "Compact excerpts. Categories are fixed; the draw within each "
       "is random at **seed 0** and uncurated. Full untruncated text for every category "
       "is browsable in `viewer.html`. Runs per the table in `README.md`.",
       ""]

# 1. one default eval pair — pos and neg twin of the SAME question
keys = sorted({(r["instruction_idx"], r["question_idx"], r["sample_idx"])
               for r in grecs if r["persona"] == "default" and r["split"] == "eval"})
i, q, k = rng.choice(keys)
for pol, cond in (("pos", "instructed-sycophantic"), ("neg", "instructed-honest")):
    rid = f"default|{pol}|eval|i{i}|q{q}|s{k}"
    r = gidx[rid]
    out.append(block("default", cond, gsc[rid], r["question"], r["response"]))

# 2. one pirate honest response
pids = sorted(r["response_id"] for r in grecs
              if r["persona"] == "pirate" and r["polarity"] == "neg" and r["split"] == "eval")
rid = rng.choice(pids)
r = gidx[rid]
out.append(block("pirate", "instructed-honest", gsc[rid], r["question"], r["response"]))

# 3. one steered response from the home sweep at coef 2.0
ssc, _, srecs = load(HOME_SWEEP)
cand = sorted((r for r in srecs if r["coef"] == 2.0), key=lambda r: r["response_id"])
r = rng.choice(cand)
out.append(block("default", "steered with v_default, coef 2.0 (α = 47.2)",
                 ssc[r["response_id"]], r["question"], r["response"]))

OUT.write_text("\n".join(out))
print(f"-> {OUT}")
