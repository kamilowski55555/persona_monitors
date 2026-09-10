#!/usr/bin/env python3
"""Open a local response panel from a responses JSON/JSONL file.

Handles both pairwise pos/neg eval runs and coefficient steering sweeps.
Scores are optional. Pass --scores now, or drop a scores file in the panel later.

Examples:
  python view_responses.py data/runs/2026-09-03-grid-s0/responses.jsonl
  python view_responses.py data/runs/2026-09-03-grid-s0/responses.jsonl --scores data/runs/2026-09-03-grid-s0/scores.jsonl
  python view_responses.py data/runs/2026-09-03-steer-pirate-s0/responses.jsonl --scores data/runs/2026-09-03-steer-pirate-s0/scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import tempfile
import webbrowser
from pathlib import Path

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Eval pairwise viewer</title>
  <style>
    :root {
      --bg: #181818;
      --fg: #e8e8e8;
      --muted: #9a9a9a;
      --dim: #6e6e6e;
      --card: #222;
      --line: #333;
      --accent: #6ea8ff;
      --pos: #7eb87e;
      --neg: #c48a5a;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      background: var(--bg);
      color: var(--fg);
      font: 15px/1.5 ui-sans-serif, system-ui, Segoe UI, sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      background: var(--bg);
      border-bottom: 1px solid var(--line);
      padding: 16px 24px 12px;
    }
    h1 { font-size: 20px; font-weight: 650; margin: 0 0 4px; }
    .sub { color: var(--muted); font-size: 13px; margin: 0 0 12px; }
    .filters {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 16px;
      align-items: end;
    }
    label {
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 11px;
      color: var(--muted);
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    select, input[type="number"] {
      background: var(--card);
      color: var(--fg);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 8px;
      font: inherit;
      min-width: 92px;
    }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; }
    .chip {
      border: 1px solid var(--line);
      background: transparent;
      color: var(--fg);
      border-radius: 999px;
      padding: 5px 10px;
      font: 13px/1 inherit;
      cursor: pointer;
    }
    .chip.on { background: #2a2a2a; border-color: #555; }
    .file-btn {
      font: 13px inherit;
      color: var(--muted);
    }
    .file-btn input { display: none; }
    .file-btn span {
      border: 1px dashed var(--line);
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
      display: inline-block;
    }
    main { padding: 20px 24px 64px; max-width: 1400px; }
    .persona { margin: 0 0 36px; }
    .persona h2 {
      font-size: 16px;
      font-weight: 650;
      margin: 0 0 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--line);
    }
    .pair { margin: 0 0 28px; }
    .question {
      font-weight: 600;
      margin: 0 0 10px;
      max-width: 90ch;
    }
    .meta {
      color: var(--dim);
      font-size: 12px;
      margin: 0 0 10px;
    }
    .cols {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      align-items: start;
    }
    @media (max-width: 900px) {
      .cols { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      min-width: 0;
    }
    .card.head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 8px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .pos .pol { color: var(--pos); }
    .neg .pol { color: var(--neg); }
    .scores { color: var(--muted); font-variant-numeric: tabular-nums; }
    .prompt {
      color: var(--dim);
      font-size: 12px;
      font-style: italic;
      margin: 0 0 10px;
    }
    .body { white-space: pre-wrap; overflow-wrap: anywhere; }
    details.prompt-box { margin: 0 0 10px; }
    details.prompt-box summary {
      cursor: pointer;
      color: var(--muted);
      font-size: 12px;
    }
    .empty { color: var(--muted); padding: 24px 0; }
    .missing {
      color: var(--dim);
      font-style: italic;
      padding: 8px 0;
    }
    .cols.steer {
      display: flex;
      overflow-x: auto;
      gap: 12px;
      padding-bottom: 4px;
    }
    .cols.steer .card {
      flex: 1 0 300px;
      max-width: 420px;
    }
    label[hidden] { display: none; }
  </style>
</head>
<body>
  <header>
    <h1 id="title">Response viewer</h1>
    <p class="sub" id="summary">No records loaded.</p>
    <div class="filters">
      <label>Split
        <select id="split"></select>
      </label>
      <label>Sample
        <select id="sample"></select>
      </label>
      <label id="instruction-wrap">Instruction
        <select id="instruction"></select>
      </label>
      <label id="limit-wrap"><span id="limit-caption">Pairs / persona</span>
        <input id="limit" type="number" min="1" max="200" value="5" />
      </label>
      <label id="offset-wrap"><span id="offset-caption">Start at pair</span>
        <input id="offset" type="number" min="0" max="200" value="0" />
      </label>
      <label>Personas
        <div class="chips" id="persona-chips"></div>
      </label>
      <label id="coef-wrap" hidden>Coefficients
        <div class="chips" id="coef-chips"></div>
      </label>
      <label class="file-btn">Responses
        <span>Load responses JSON<input id="responses-file" type="file" accept=".json,.jsonl,application/json" /></span>
      </label>
      <label class="file-btn">Scores
        <span>Load scores JSON<input id="scores-file" type="file" accept=".json,.jsonl,application/json" /></span>
      </label>
    </div>
  </header>
  <main id="main"></main>
  <script id="boot-data" type="application/json">%%DATA%%</script>
  <script>
    const $ = (id) => document.getElementById(id);
    let records = [];
    let scores = {};
    let selectedPersonas = new Set();
    let selectedCoefs = new Set();

    function isSteerMode() {
      return records.some((r) => r && (r.coef != null || r.polarity === "steer"));
    }

    function sortedCoefs() {
      return unique(records.map((r) => r.coef)).sort((a, b) => a - b);
    }

    function parseRecords(text) {
      const trimmed = text.trim();
      if (!trimmed) return [];
      if (trimmed.startsWith("[")) {
        const data = JSON.parse(trimmed);
        return Array.isArray(data) ? data : [data];
      }
      const out = [];
      for (const line of trimmed.split(/\r?\n/)) {
        if (line.trim()) out.push(JSON.parse(line));
      }
      return out;
    }

    function indexScores(rows) {
      const map = {};
      for (const row of rows) {
        if (row && row.response_id) map[row.response_id] = row;
      }
      return map;
    }

    function scored(row) {
      const extra = row.response_id ? scores[row.response_id] : null;
      return {
        ...row,
        trait_score: extra?.trait_score ?? row.trait_score,
        coherence_score: extra?.coherence_score ?? row.coherence_score,
        trait_mass: extra?.trait_mass ?? row.trait_mass,
        coherence_mass: extra?.coherence_mass ?? row.coherence_mass,
      };
    }

    function unique(values) {
      return [...new Set(values.filter((v) => v !== undefined && v !== null))];
    }

    function fmt(n, d = 1) {
      return typeof n === "number" && Number.isFinite(n) ? n.toFixed(d) : null;
    }

    function fillSelect(el, values, format) {
      const prev = el.value;
      el.innerHTML = "";
      for (const value of values) {
        const opt = document.createElement("option");
        opt.value = String(value);
        opt.textContent = format ? format(value) : String(value);
        el.appendChild(opt);
      }
      const wanted = [...el.options].some((o) => o.value === prev) ? prev : el.options[0]?.value;
      if (wanted !== undefined) el.value = wanted;
    }

    function fillChips(el, values, selected, onToggle, format) {
      el.innerHTML = "";
      if (selected.size === 0) values.forEach((v) => selected.add(v));
      for (const value of values) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "chip" + (selected.has(value) ? " on" : "");
        btn.textContent = format ? format(value) : String(value);
        btn.addEventListener("click", () => {
          if (selected.has(value) && selected.size > 1) selected.delete(value);
          else selected.add(value);
          onToggle();
        });
        el.appendChild(btn);
      }
    }

    function rebuildFilters() {
      const steer = isSteerMode();
      const splits = unique(records.map((r) => r.split)).sort();
      const samples = unique(records.map((r) => r.sample_idx)).sort((a, b) => a - b);
      const instructions = unique(records.map((r) => r.instruction_idx)).sort((a, b) => a - b);
      const personas = unique(records.map((r) => r.persona));
      const coefs = sortedCoefs();
      personas.sort((a, b) => {
        if (a === "default") return -1;
        if (b === "default") return 1;
        return a.localeCompare(b);
      });

      $("title").textContent = steer ? "Steer sweep viewer" : "Eval pairwise viewer";
      $("instruction-wrap").hidden = steer;
      $("coef-wrap").hidden = !steer;
      $("limit-caption").textContent = steer ? "Questions / persona" : "Pairs / persona";
      $("offset-caption").textContent = steer ? "Start at question" : "Start at pair";

      fillSelect($("split"), splits.length ? splits : ["eval"]);
      if (splits.includes("eval")) $("split").value = "eval";
      else if (splits.length) $("split").value = String(splits[0]);
      fillSelect($("sample"), samples.length ? samples : [0], (v) => "s" + v);
      fillSelect($("instruction"), instructions.length ? instructions : [0], (v) => "i" + v);

      fillChips($("persona-chips"), personas, selectedPersonas, () => {
        rebuildFilters();
        render();
      });
      if (steer) {
        fillChips($("coef-chips"), coefs, selectedCoefs, () => {
          rebuildFilters();
          render();
        }, (v) => "c" + v);
      }
    }

    function pairKey(row) {
      return [row.persona, row.split, row.instruction_idx, row.question_idx, row.sample_idx].join("|");
    }

    function steerKey(row) {
      return [row.persona, row.split, row.question_idx, row.sample_idx].join("|");
    }

    function groupByPersona(grouped) {
      const byPersona = new Map();
      for (const item of grouped.values()) {
        if (!byPersona.has(item.persona)) byPersona.set(item.persona, []);
        byPersona.get(item.persona).push(item);
      }
      for (const list of byPersona.values()) {
        list.sort((a, b) => a.question_idx - b.question_idx);
      }
      return byPersona;
    }

    function buildPairs() {
      const split = $("split").value;
      const sample = Number($("sample").value);
      const instruction = Number($("instruction").value);
      const grouped = new Map();
      for (const raw of records) {
        const row = scored(raw);
        if (String(row.split) !== split) continue;
        if (Number(row.sample_idx) !== sample) continue;
        if (Number(row.instruction_idx) !== instruction) continue;
        if (!selectedPersonas.has(row.persona)) continue;
        const key = pairKey(row);
        if (!grouped.has(key)) {
          grouped.set(key, {
            persona: row.persona,
            split: row.split,
            instruction_idx: row.instruction_idx,
            question_idx: row.question_idx,
            sample_idx: row.sample_idx,
            question: row.question || "",
            pos: null,
            neg: null,
          });
        }
        const pair = grouped.get(key);
        if (row.polarity === "pos") pair.pos = row;
        if (row.polarity === "neg") pair.neg = row;
        if (row.question) pair.question = row.question;
      }
      return groupByPersona(grouped);
    }

    function buildSteerGroups() {
      const split = $("split").value;
      const sample = Number($("sample").value);
      const grouped = new Map();
      for (const raw of records) {
        const row = scored(raw);
        if (String(row.split ?? "steer") !== split) continue;
        if (Number(row.sample_idx) !== sample) continue;
        if (!selectedPersonas.has(row.persona)) continue;
        if (row.coef != null && !selectedCoefs.has(row.coef)) continue;
        const key = steerKey(row);
        if (!grouped.has(key)) {
          grouped.set(key, {
            persona: row.persona,
            split: row.split,
            question_idx: row.question_idx,
            sample_idx: row.sample_idx,
            question: row.question || "",
            byCoef: new Map(),
          });
        }
        const group = grouped.get(key);
        if (row.coef != null) group.byCoef.set(row.coef, row);
        if (row.question) group.question = row.question;
      }
      return groupByPersona(grouped);
    }

    function scoreLine(row) {
      if (!row) return "";
      const t = fmt(row.trait_score);
      const c = fmt(row.coherence_score);
      if (t === null && c === null) return "";
      const bits = [];
      if (t !== null) bits.push("T " + t);
      if (c !== null) bits.push("C " + c);
      return bits.join(" · ");
    }

    function card(label, row, extraClass, missingText) {
      const el = document.createElement("article");
      el.className = "card " + (extraClass || "");
      const head = document.createElement("div");
      head.className = "head";
      const pol = document.createElement("span");
      pol.className = "pol";
      pol.textContent = label;
      const scoresEl = document.createElement("span");
      scoresEl.className = "scores";
      scoresEl.textContent = scoreLine(row);
      head.append(pol, scoresEl);
      el.appendChild(head);
      if (!row) {
        const missing = document.createElement("p");
        missing.className = "missing";
        missing.textContent = missingText || ("No matching " + label + " response.");
        el.appendChild(missing);
        return el;
      }
      if (row.system_prompt) {
        const details = document.createElement("details");
        details.className = "prompt-box";
        const summary = document.createElement("summary");
        summary.textContent = "system prompt";
        const prompt = document.createElement("p");
        prompt.className = "prompt";
        prompt.textContent = row.system_prompt;
        details.append(summary, prompt);
        el.appendChild(details);
      }
      const body = document.createElement("div");
      body.className = "body";
      body.textContent = row.response || "";
      el.appendChild(body);
      return el;
    }

    function steerLabel(row, coef) {
      const c = row?.coef ?? coef;
      const a = fmt(row?.alpha, 1);
      return a ? ("c" + c + " · α " + a) : ("c" + c);
    }

    function renderPairwise(byPersona, limit, offset) {
      const main = $("main");
      let shown = 0;
      let scoredCount = 0;
      let pairCount = 0;
      for (const [persona, pairs] of byPersona) {
        pairCount += pairs.length;
        const slice = pairs.slice(offset, offset + limit);
        const section = document.createElement("section");
        section.className = "persona";
        const h2 = document.createElement("h2");
        h2.textContent = persona + "  ·  " + slice.length + " of " + pairs.length + " pairs";
        section.appendChild(h2);
        if (!slice.length) {
          const empty = document.createElement("p");
          empty.className = "empty";
          empty.textContent = "No pairs in this window. Lower the start offset or raise the pair count.";
          section.appendChild(empty);
        }
        for (const pair of slice) {
          shown += 1;
          const wrap = document.createElement("article");
          wrap.className = "pair";
          const q = document.createElement("p");
          q.className = "question";
          q.textContent = pair.question;
          const meta = document.createElement("p");
          meta.className = "meta";
          meta.textContent = "q" + pair.question_idx + " · i" + pair.instruction_idx + " · s" + pair.sample_idx + " · " + pair.split;
          const cols = document.createElement("div");
          cols.className = "cols";
          cols.append(
            card("pos · sycophantic", pair.pos, "pos", "No matching pos response for this pair."),
            card("neg · honest", pair.neg, "neg", "No matching neg response for this pair."),
          );
          if (scoreLine(pair.pos) || scoreLine(pair.neg)) scoredCount += 1;
          wrap.append(q, meta, cols);
          section.appendChild(wrap);
        }
        main.appendChild(section);
      }
      return { shown, scoredCount, groupCount: pairCount, noun: "pairs" };
    }

    function renderSteer(byPersona, limit, offset) {
      const main = $("main");
      const coefs = sortedCoefs().filter((c) => selectedCoefs.has(c));
      let shown = 0;
      let scoredCount = 0;
      let groupCount = 0;
      for (const [persona, groups] of byPersona) {
        groupCount += groups.length;
        const slice = groups.slice(offset, offset + limit);
        const section = document.createElement("section");
        section.className = "persona";
        const h2 = document.createElement("h2");
        h2.textContent = persona + "  ·  " + slice.length + " of " + groups.length + " questions";
        section.appendChild(h2);
        if (!slice.length) {
          const empty = document.createElement("p");
          empty.className = "empty";
          empty.textContent = "No questions in this window. Lower the start offset or raise the question count.";
          section.appendChild(empty);
        }
        for (const group of slice) {
          shown += 1;
          const wrap = document.createElement("article");
          wrap.className = "pair";
          const q = document.createElement("p");
          q.className = "question";
          q.textContent = group.question;
          const meta = document.createElement("p");
          meta.className = "meta";
          meta.textContent = "q" + group.question_idx + " · s" + group.sample_idx + " · " + (group.split || "steer");
          const cols = document.createElement("div");
          cols.className = "cols steer";
          let anyScore = false;
          for (const coef of coefs) {
            const row = group.byCoef.get(coef) || null;
            cols.appendChild(card(
              steerLabel(row, coef),
              row,
              "",
              "No matching response at c" + coef + ".",
            ));
            if (scoreLine(row)) anyScore = true;
          }
          if (anyScore) scoredCount += 1;
          wrap.append(q, meta, cols);
          section.appendChild(wrap);
        }
        main.appendChild(section);
      }
      return { shown, scoredCount, groupCount, noun: "questions" };
    }

    function render() {
      const main = $("main");
      main.innerHTML = "";
      if (!records.length) {
        main.innerHTML = '<p class="empty">Load a responses JSON/JSONL file to start.</p>';
        $("summary").textContent = "No records loaded.";
        return;
      }
      const steer = isSteerMode();
      const limit = Math.max(1, Number($("limit").value) || 5);
      const offset = Math.max(0, Number($("offset").value) || 0);
      const stats = steer
        ? renderSteer(buildSteerGroups(), limit, offset)
        : renderPairwise(buildPairs(), limit, offset);
      if (!stats.groupCount && !main.children.length) {
        main.innerHTML = '<p class="empty">No ' + stats.noun + " match these filters.</p>";
      }
      const scoreNote = Object.keys(scores).length || records.some((r) => r.trait_score != null)
        ? stats.scoredCount + " shown " + stats.noun + " have scores"
        : "no scores loaded yet — use Load scores JSON when you have them";
      $("summary").textContent =
        records.length + " responses  ·  " + stats.groupCount + " " + stats.noun + " in view  ·  showing " + stats.shown + "  ·  " + scoreNote;
    }

    function boot() {
      const boot = JSON.parse($("boot-data").textContent);
      records = boot.records || [];
      scores = indexScores(boot.scores || []);
      $("limit").value = String(boot.limit || 5);
      $("offset").value = "0";
      rebuildFilters();
      if (boot.split) $("split").value = boot.split;
      if (boot.sample != null && [...$("sample").options].some((o) => o.value === String(boot.sample))) {
        $("sample").value = String(boot.sample);
      }
      if (boot.instruction != null && [...$("instruction").options].some((o) => o.value === String(boot.instruction))) {
        $("instruction").value = String(boot.instruction);
      }
      ["split", "sample", "instruction", "limit", "offset"].forEach((id) => {
        $(id).addEventListener("change", render);
        $(id).addEventListener("input", render);
      });
      $("responses-file").addEventListener("change", async (ev) => {
        const file = ev.target.files?.[0];
        if (!file) return;
        records = parseRecords(await file.text());
        selectedPersonas = new Set();
        selectedCoefs = new Set();
        rebuildFilters();
        render();
      });
      $("scores-file").addEventListener("change", async (ev) => {
        const file = ev.target.files?.[0];
        if (!file) return;
        scores = indexScores(parseRecords(await file.text()));
        render();
      });
      render();
    }

    boot();
  </script>
</body>
</html>
"""


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def infer_split(records: list[dict], requested: str | None) -> str:
    if requested:
        return requested
    splits = []
    seen: set[object] = set()
    for row in records:
        value = row.get("split")
        if value is None or value in seen:
            continue
        seen.add(value)
        splits.append(value)
    if "eval" in splits:
        return "eval"
    if splits:
        return str(splits[0])
    return "eval"


def embed_json(value: object) -> str:
    dumped = json.dumps(value, ensure_ascii=False)
    return dumped.replace("<", "\\u003c")


def build_html(
    records: list[dict],
    scores: list[dict],
    *,
    limit: int,
    split: str,
    sample: int,
    instruction: int,
) -> str:
    payload = {
        "records": records,
        "scores": scores,
        "limit": limit,
        "split": split,
        "sample": sample,
        "instruction": instruction,
    }
    return TEMPLATE.replace("%%DATA%%", embed_json(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse pairwise eval or steering-sweep responses.")
    parser.add_argument("responses", type=Path, help="Responses JSON or JSONL")
    parser.add_argument("--scores", type=Path, help="Optional scores JSON/JSONL joined on response_id")
    parser.add_argument("--limit", type=int, default=5, help="Pairs/questions to show per persona (default: 5)")
    parser.add_argument("--split", default=None, help="Initial split filter (default: eval if present, else first split)")
    parser.add_argument("--sample", type=int, default=0, help="Initial sample_idx (default: 0 = s0)")
    parser.add_argument("--instruction", type=int, default=0, help="Initial instruction_idx for pairwise runs (default: 0)")
    parser.add_argument("--out", type=Path, help="Write HTML here instead of a temp file")
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser")
    args = parser.parse_args()

    records = load_records(args.responses)
    scores = load_records(args.scores) if args.scores else []
    split = infer_split(records, args.split)
    html = build_html(
        records,
        scores,
        limit=args.limit,
        split=split,
        sample=args.sample,
        instruction=args.instruction,
    )

    out = args.out
    if out is None:
        handle = tempfile.NamedTemporaryFile(
            prefix="eval-panel-",
            suffix=".html",
            delete=False,
            mode="w",
            encoding="utf-8",
        )
        out = Path(handle.name)
        handle.write(html)
        handle.close()
    else:
        out.write_text(html, encoding="utf-8")

    mode = "steer" if any(r.get("coef") is not None or r.get("polarity") == "steer" for r in records) else "pairwise"
    print(f"Wrote {out}")
    print(f"{len(records)} responses ({mode}), {len(scores)} score rows, split={split}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
