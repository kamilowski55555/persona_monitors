"""Paired kept-response filter + elicitation report (a persona
where elicitation fails dies HERE, loudly). Filter copied from persona_vectors
generate_vec.py L40-43. Usage: .venv/bin/python -m src.filter_report data/runs/<run_id>"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from src.config import (FILTER_THRESHOLD, COHERENCE_MIN, KILL_MIN_PAIRS,
                        KILL_MIN_FRAC_OF_DEFAULT)

# CJK ideographs + kana: Qwen code-switches under temperature 1.0. Counted, never filtered.
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")


def load_pairs(run_dir):
    """(persona, split, instr, question, sample) -> {polarity: score row}. Shared by
    filter_report, vectors and metrics — the pairing convention lives here only."""
    scores = {s["response_id"]: s for s in map(json.loads, open(run_dir / "scores.jsonl"))}
    pairs = defaultdict(dict)
    for r in map(json.loads, open(run_dir / "responses.jsonl")):
        key = (r["persona"], r["split"], r["instruction_idx"], r["question_idx"], r["sample_idx"])
        pairs[key][r["polarity"]] = scores[r["response_id"]]
    return pairs


def passes_filter(pair):
    """persona_vectors generate_vec.py L40-43, on an aligned pos/neg pair.
    Single definition — every module that filters calls this one."""
    if len(pair) != 2:
        return False
    if any(pair[p][k] is None for p in ("pos", "neg") for k in ("trait_score", "coherence_score")):
        return False
    return (pair["pos"]["trait_score"] >= FILTER_THRESHOLD
            and pair["neg"]["trait_score"] < 100 - FILTER_THRESHOLD
            and pair["pos"]["coherence_score"] >= COHERENCE_MIN
            and pair["neg"]["coherence_score"] >= COHERENCE_MIN)


def kept_pairs(run_dir, persona=None, split=None, use_filter=True):
    """Surviving pairs as (key, pair) list, optionally restricted to one condition.
    use_filter=False returns every complete, non-refused pair instead — the filter-bias
    check (vectors.py --no-filter), never the default path."""
    def ok(pair):
        if use_filter:
            return passes_filter(pair)
        return len(pair) == 2 and not any(
            pair[q][k] is None for q in ("pos", "neg") for k in ("trait_score", "coherence_score"))
    return [(k, p) for k, p in load_pairs(run_dir).items() if ok(p)
            and (persona is None or k[0] == persona) and (split is None or k[1] == split)]


def main(run_dir):
    pairs = load_pairs(run_dir)
    scores = {s["response_id"]: s for s in map(json.loads, open(run_dir / "scores.jsonl"))}
    cjk = defaultdict(int)  # code-switch audit: responses containing CJK, per condition
    for r in map(json.loads, open(run_dir / "responses.jsonl")):
        if CJK_RE.search(r["response"]):
            cjk[(r["persona"], r["polarity"], r["split"])] += 1

    # per (persona, polarity, split) score distributions — makes a persona-level score
    # shift visible here rather than downstream (smoke: pirate inflated in BOTH polarities)
    dist = defaultdict(lambda: dict(trait=[], coherence=[]))
    for s in scores.values():
        d = dist[(s["persona"], s["polarity"], s["split"])]
        for k, col in (("trait", "trait_score"), ("coherence", "coherence_score")):
            if s[col] is not None:
                d[k].append(s[col])

    stat = defaultdict(lambda: dict(raw=0, refusal_pairs=0, refusal_responses=0, kept=0))
    for (persona, split, *_), pair in pairs.items():
        st = stat[(persona, split)]
        if len(pair) != 2:
            continue  # unpaired rows would be a generation bug; counted nowhere but visible as raw mismatch
        st["raw"] += 1
        n_null = sum(pair[p][k] is None for p in ("pos", "neg") for k in ("trait_score", "coherence_score"))
        if n_null:
            st["refusal_pairs"] += 1
            st["refusal_responses"] += n_null
            continue  # excluded from the filter, recorded + counted, never silently dropped
        if passes_filter(pair):
            st["kept"] += 1

    print(f"{'persona':<12}{'split':<9}{'raw':>5}{'kept':>6}{'surv%':>7}{'refusal_pairs':>14}{'refusal_resp':>13}")
    for (persona, split), st in sorted(stat.items()):
        pct = 100 * st["kept"] / st["raw"] if st["raw"] else 0.0
        print(f"{persona:<12}{split:<9}{st['raw']:>5}{st['kept']:>6}{pct:>6.1f}%{st['refusal_pairs']:>14}{st['refusal_responses']:>13}")

    print(f"\n{'persona':<12}{'pol':<5}{'split':<9}{'n':>4}{'trait_mean':>12}{'trait_med':>11}{'coh_mean':>10}{'cjk_resp':>10}")
    for (persona, polarity, split), d in sorted(dist.items()):
        t, c = d["trait"], d["coherence"]
        tm = f"{statistics.mean(t):.1f}" if t else "-"
        tmed = f"{statistics.median(t):.1f}" if t else "-"
        cm = f"{statistics.mean(c):.1f}" if c else "-"
        print(f"{persona:<12}{polarity:<5}{split:<9}{len(t):>4}{tm:>12}{tmed:>11}{cm:>10}"
              f"{cjk[(persona, polarity, split)]:>10}")
    print("(a persona shifted in BOTH polarities suggests judge style contamination, "
          "not elicitation — quantify at pilot scale)")
    print("(cjk_resp = responses containing CJK characters — temp-1.0 code-switch audit, "
          "counted not filtered)\n")

    for (persona, split), st in sorted(stat.items()):  # pre-registered elicitation kill criteria
        default_kept = stat.get(("default", split), {"kept": 0})["kept"]
        floor = KILL_MIN_FRAC_OF_DEFAULT * default_kept
        if st["kept"] < KILL_MIN_PAIRS:
            print(f"KILL FLAG: {persona}/{split} kept {st['kept']} < KILL_MIN_PAIRS={KILL_MIN_PAIRS}")
        if persona != "default" and default_kept and st["kept"] < floor:
            print(f"KILL FLAG: {persona}/{split} kept {st['kept']} < {KILL_MIN_FRAC_OF_DEFAULT} x default ({default_kept})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    main(ap.parse_args().run_dir)
