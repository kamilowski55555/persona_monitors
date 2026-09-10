"""Emit viewer.html — a self-contained browser over the canonical runs.

Two modes, mirroring how the data is actually shaped:
  * Eval pairwise — the pos/neg twins of one question, side by side.
  * Steer sweep   — one question, every dose across the row, so the ramp reads L->R.

A working subset is baked in so the page is useful with zero setup: the complete steering
sweeps, plus the grid at instruction i0 / sample s0 for both splits. The full grid is
24,000 responses / 40 MB, far too large to inline, so the rest is reachable through the
LOAD JSONL pickers — files are read in the browser, nothing is uploaded.

Grouping note: persona alone does NOT identify a steering sweep. `pirate` was swept twice,
once with v_default and once with v_pirate at matched absolute alpha, so the selector is
keyed on the sweep (persona x vector), never on the persona.

  .venv/bin/python make_viewer.py
"""

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.config import PERSONAS_JSON
from src.generate import a_or_an

GRID = Path("data/runs/2026-09-03-grid-s0")
SWEEPS = [Path("data/runs/2026-09-03-steer-default-s0"),
          Path("data/runs/2026-09-03-steer-pirate-s0"),
          Path("data/runs/2026-09-04-steer-pirate-v_pirate-absA-s0")]
OUT = Path("viewer.html")
BAKED_SAMPLES = (0,)        # grid sample indices baked in; the rest load from JSONL
BAKED_INSTR = (0,)          # grid instruction indices baked in

qs, sps = {}, {}            # dedup tables: text -> index


def qi(t):
    return qs.setdefault(t, len(qs))


def si(t):
    return sps.setdefault(t, len(sps))


# Steering runs record no system_prompt, so reconstruct the one src/steering.py composed
# (L81-82). The guard trips if that line is ever edited, rather than letting the viewer
# quietly show a stale prompt.
STEER_SYS_TMPL = 'f"You are {a_or_an(noun)} {noun}; stay fully in character as {a_or_an(noun)} {noun}."'
assert STEER_SYS_TMPL in Path("src/steering.py").read_text(), \
    "src/steering.py's persona system prompt changed — update STEER_SYS_TMPL here to match."
NOUNS = {p["id"]: p["noun"] for p in json.load(open(PERSONAS_JSON))["personas"]}


def steer_system(persona):
    noun = NOUNS[persona]
    if not noun:  # default persona: steering passes no system message at all
        return "(none — no system message, matching persona_vectors' steering eval)"
    return f"You are {a_or_an(noun)} {noun}; stay fully in character as {a_or_an(noun)} {noun}."


def load(run_dir):
    sc = {s["response_id"]: s for s in map(json.loads, open(run_dir / "scores.jsonl"))}
    return sc, [json.loads(l) for l in open(run_dir / "responses.jsonl")]


# ---- eval / extract: baked slice of the grid ---------------------------------------
gsc, grecs = load(GRID)
ev = []
for r in grecs:
    if r["instruction_idx"] in BAKED_INSTR and r["sample_idx"] in BAKED_SAMPLES:
        s = gsc[r["response_id"]]
        ev.append({"p": r["persona"], "pol": r["polarity"], "sl": r["split"],
                   "i": r["instruction_idx"], "q": r["question_idx"], "s": r["sample_idx"],
                   "sp": si(r["system_prompt"]), "qt": qi(r["question"]),
                   "r": r["response"], "run": GRID.name,
                   "t": s["trait_score"], "c": s["coherence_score"]})

# ---- steer: all three sweeps, complete ----------------------------------------------
st = []
for sd in SWEEPS:
    ssc, srecs = load(sd)
    meta = json.loads((sd / "run_meta.json").read_text())
    vk = meta.get("vector_key", "v_default")
    sys_p = si(steer_system(meta["persona"]))
    # the sweep label must distinguish pirate/v_default from pirate/v_pirate
    sw = f"{meta['persona']} · {vk}" + (" · matched α" if "absA" in sd.name else "")
    for r in srecs:
        s = ssc[r["response_id"]]
        st.append({"sw": sw, "p": meta["persona"], "coef": r["coef"],
                   "a": round(r["alpha"], 1), "q": r["question_idx"],
                   "s": r["sample_idx"], "qt": qi(r["question"]), "sp": sys_p,
                   "r": r["response"], "run": sd.name,
                   "t": s["trait_score"], "c": s["coherence_score"]})

payload = json.dumps({"ev": ev, "st": st,
                      "Q": [k for k, _ in sorted(qs.items(), key=lambda kv: kv[1])],
                      "SP": [k for k, _ in sorted(sps.items(), key=lambda kv: kv[1])]},
                     ensure_ascii=False, separators=(",", ":"))

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
runs = ", ".join([GRID.name] + [s.name for s in SWEEPS])

OUT.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>persona-monitors — response viewer</title>
<style>
:root {{ color-scheme: dark; --bg:#1a1a19; --panel:#232322; --line:#3a3a38; --fg:#e8e6e1;
  --dim:#9a978f; --pos:#d98b6b; --neg:#7fb069; --accent:#d8c99b; }}
*{{box-sizing:border-box}}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
.mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
header.top {{ padding:14px 18px 0; }}
h1 {{ font-size:19px; margin:0 0 4px; }}
.counts {{ color:var(--dim); font-size:12.5px; margin:0 0 10px; }}
.bar {{ display:flex; flex-wrap:wrap; gap:14px; align-items:flex-end;
  border-bottom:1px solid var(--line); padding:0 18px 12px; }}
.grp {{ display:flex; flex-direction:column; gap:4px; }}
.grp>label {{ font-size:10.5px; letter-spacing:.06em; color:var(--dim); text-transform:uppercase; }}
select,input[type=number] {{ background:var(--panel); color:var(--fg); border:1px solid var(--line);
  border-radius:5px; padding:5px 7px; font:inherit; font-size:13px; min-width:70px; }}
.chips {{ display:flex; gap:5px; flex-wrap:wrap; }}
.chip {{ background:var(--panel); color:var(--dim); border:1px solid var(--line); cursor:pointer;
  border-radius:999px; padding:4px 11px; font:inherit; font-size:12.5px; }}
.chip[aria-pressed=true] {{ background:var(--fg); color:var(--bg); border-color:var(--fg); }}
button.file {{ background:var(--panel); color:var(--fg); border:1px solid var(--line);
  border-radius:5px; padding:6px 10px; font:inherit; font-size:11.5px; cursor:pointer;
  letter-spacing:.04em; }}
main {{ padding:14px 18px 60px; }}
.pgroup>h2 {{ font-size:15px; margin:22px 0 6px; border-bottom:1px solid var(--line);
  padding-bottom:5px; }}
.qblock {{ margin:0 0 26px; }}
.qtext {{ font-weight:600; margin:14px 0 2px; }}
.qmeta {{ color:var(--dim); font-size:11.5px; margin-bottom:8px; }}
.row {{ display:flex; gap:10px; align-items:stretch; }}
.row.scroll {{ overflow-x:auto; padding-bottom:6px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:7px;
  padding:10px 12px; min-width:0; }}
.row .card {{ flex:1 1 0; }}
.row.scroll .card {{ flex:0 0 300px; }}
.chead {{ font-size:11.5px; color:var(--dim); margin-bottom:7px; }}
.chead b {{ color:var(--fg); font-weight:600; }}
.pos b {{ color:var(--pos); }} .neg b {{ color:var(--neg); }}
details {{ margin:0 0 7px; }}
summary {{ cursor:pointer; color:var(--dim); font-size:11.5px; }}
details p {{ white-space:pre-wrap; color:var(--dim); font-size:12.5px;
  border-left:2px solid var(--line); padding-left:8px; margin:6px 0 0; }}
.body {{ white-space:pre-wrap; font-size:14px; }}
.note {{ color:var(--dim); font-size:12.5px; max-width:860px; }}
.note a {{ color:var(--accent); }}
@media (max-width:760px) {{ .row {{ display:block; }} .row .card {{ margin-bottom:10px; }} }}
</style></head><body>
<header class="top">
<h1 id="title">Pairwise viewer</h1>
<p class="counts" id="counts"></p>
</header>
<div class="bar">
  <div class="grp"><label>Mode</label><select id="mode">
    <option value="ev">Pairwise</option><option value="st">Steer sweep</option></select></div>
  <div class="grp" id="g-split"><label>Split</label><select id="split"></select></div>
  <div class="grp" id="g-instr"><label>Instruction</label><select id="instr"></select></div>
  <div class="grp" id="g-sweep"><label>Sweep</label><select id="sweep"></select></div>
  <div class="grp"><label>Sample</label><select id="samp"></select></div>
  <div class="grp"><label id="lab-n">Questions / persona</label>
    <input type="number" id="npp" value="5" min="1" max="200"></div>
  <div class="grp"><label>Start at</label>
    <input type="number" id="start" value="0" min="0"></div>
  <div class="grp" id="g-pers"><label>Personas</label><div class="chips" id="chip-p"></div></div>
  <div class="grp" id="g-coef"><label>Doses (α)</label><div class="chips" id="chip-c"></div></div>
  <div class="grp"><label>Responses</label>
    <button class="file" onclick="pick('resp')">LOAD RESPONSES JSONL</button></div>
  <div class="grp"><label>Scores</label>
    <button class="file" onclick="pick('score')">LOAD SCORES JSONL</button></div>
</div>
<main>
<p class="note" id="intro">Baked in: the three steering sweeps in full, and the grid at
instruction i0 / sample s0 for both splits. The complete grid is 24,000 responses (40&nbsp;MB) —
too large to inline — so to browse the rest, load a run's
<code>responses.jsonl</code> and then its <code>scores.jsonl</code> with the buttons above.
Files are parsed in your browser; nothing is uploaded. Nothing here is curated: the controls
select, they never rank.</p>
<div id="out"></div>
<p class="note mono" style="margin-top:26px">generated {stamp} · runs: {runs}</p>
</main>
<input type="file" id="fp" accept=".jsonl,.json" hidden>
<script>
const D = {payload};
let EV = D.ev, ST = D.st, Q = D.Q, SP = D.SP;
let loadedResp = null, selP = new Set(), selC = new Set(), mode = "ev";
const $ = id => document.getElementById(id);
const uniq = (a, f) => [...new Set(a.map(f))].sort(
  (x, y) => (typeof x === "number" ? x - y : String(x).localeCompare(String(y))));
const esc = s => {{ const d = document.createElement("div"); d.textContent = s == null ? "" : s; return d.innerHTML; }};
const num = v => (v == null || isNaN(v)) ? "—" : (+v).toFixed(1);

function chips(host, vals, store, fmt) {{
  host.innerHTML = "";
  vals.forEach(v => {{
    const b = document.createElement("button");
    b.className = "chip"; b.textContent = fmt ? fmt(v) : v;
    b.setAttribute("aria-pressed", store.has(v));
    b.onclick = () => {{
      store.has(v) ? store.delete(v) : store.add(v);
      b.setAttribute("aria-pressed", store.has(v));
      render();
    }};
    host.appendChild(b);
  }});
}}

function opts(el, vals, fmt) {{
  const keep = el.value;
  el.innerHTML = vals.map(v => `<option value="${{esc(v)}}">${{fmt ? fmt(v) : esc(v)}}</option>`).join("");
  if (vals.map(String).includes(keep)) el.value = keep;
}}

function card(r, cls, head) {{
  const sys = r.sp != null && SP[r.sp]
    ? `<details><summary>system prompt</summary><p>${{esc(SP[r.sp])}}</p></details>` : "";
  return `<div class="card ${{cls || ""}}"><div class="chead mono">${{head}}</div>`
       + sys + `<div class="body">${{esc(r.r)}}</div></div>`;
}}

/* Rebuild every control from the data currently in memory. */
function setup() {{
  const steer = mode === "st";
  $("title").textContent = steer ? "Steer sweep viewer" : "Pairwise viewer";
  $("lab-n").textContent = steer ? "Questions / persona" : "Pairs / persona";
  $("g-split").style.display = steer ? "none" : "";
  $("g-instr").style.display = steer ? "none" : "";
  $("g-pers").style.display  = steer ? "none" : "";
  $("g-sweep").style.display = steer ? "" : "none";
  $("g-coef").style.display  = steer ? "" : "none";

  if (steer) {{
    opts($("sweep"), uniq(ST, r => r.sw));
    syncSweep();
  }} else {{
    opts($("split"), uniq(EV, r => r.sl));
    opts($("instr"), uniq(EV, r => r.i), v => "i" + v);
    const ps = uniq(EV, r => r.p);
    if (![...selP].some(p => ps.includes(p)))
      selP = new Set([ps.includes("default") ? "default" : ps[0]]);
    chips($("chip-p"), ps, selP);
    opts($("samp"), uniq(EV, r => r.s), v => "s" + v);
    render();
  }}
}}

/* Doses belong to a sweep: pirate/v_default and pirate/v_pirate use different coefs. */
function syncSweep() {{
  const rows = ST.filter(r => r.sw === $("sweep").value);
  const alphas = uniq(rows, r => r.a);
  selC = new Set(alphas);
  chips($("chip-c"), alphas, selC, a => "α " + num(a));
  opts($("samp"), uniq(rows, r => r.s), v => "s" + v);
  render();
}}

function render() {{
  const steer = mode === "st";
  const s = +$("samp").value, n = +$("npp").value, start = +$("start").value;
  let view, groups;
  if (steer) {{
    const sw = $("sweep").value;
    view = ST.filter(r => r.sw === sw && r.s === s && selC.has(r.a));
    groups = [sw];
  }} else {{
    const sl = $("split").value, instr = +$("instr").value;
    view = EV.filter(r => r.sl === sl && r.i === instr && r.s === s && selP.has(r.p));
    groups = [...selP].sort();
  }}
  const qsIn = uniq(view, r => r.q);
  const shownQ = qsIn.slice(start, start + n);
  const noun = steer ? "questions" : "pairs";

  let html = "", shown = 0, scored = 0;
  for (const g of groups) {{
    const mine = steer ? view : view.filter(r => r.p === g);
    if (!mine.length) continue;
    const mineQ = shownQ.filter(q => mine.some(r => r.q === q));
    html += `<section class="pgroup"><h2>${{esc(g)}} · ${{mineQ.length}} of ${{qsIn.length}} ${{noun}}</h2>`;
    for (const q of mineQ) {{
      const grp = mine.filter(r => r.q === q);
      shown++;
      if (grp.every(r => r.t != null)) scored++;
      const meta = steer ? `q${{q}} · s${{s}} · ${{grp[0].run}}`
                         : `q${{q}} · i${{$("instr").value}} · s${{s}} · ${{grp[0].run}}`;
      html += `<div class="qblock"><div class="qtext">${{esc(Q[grp[0].qt])}}</div>`
            + `<div class="qmeta mono">${{meta}}</div>`;
      if (steer) {{
        html += `<div class="row scroll">` + grp.sort((a, b) => a.a - b.a).map(r =>
          card(r, "", `<b>α ${{num(r.a)}}</b> · coef ${{(+r.coef).toFixed(2)}}`
                    + ` &nbsp; T ${{num(r.t)}} · C ${{num(r.c)}}`)).join("") + `</div>`;
      }} else {{
        const pos = grp.find(r => r.pol === "pos"), neg = grp.find(r => r.pol === "neg");
        html += `<div class="row">`
          + (pos ? card(pos, "pos", `<b>pos · sycophantic</b> T ${{num(pos.t)}} · C ${{num(pos.c)}}`) : "")
          + (neg ? card(neg, "neg", `<b>neg · honest</b> T ${{num(neg.t)}} · C ${{num(neg.c)}}`) : "")
          + `</div>`;
      }}
      html += `</div>`;
    }}
    html += `</section>`;
  }}
  $("out").innerHTML = html || `<p class="note">Nothing matches these filters.</p>`;
  $("counts").textContent =
    `${{(steer ? ST : EV).length.toLocaleString()}} responses · ${{qsIn.length}} ${{noun}} in view`
    + ` · showing ${{shown}} · ${{scored}} shown ${{noun}} have scores`;
}}

/* Load a full run from local disk. Responses first, then the matching scores. */
function pick(kind) {{
  const fp = $("fp");
  fp.onchange = () => {{
    const f = fp.files[0];
    if (!f) return;
    const rd = new FileReader();
    rd.onload = () => {{
      let rows;
      try {{
        rows = rd.result.split("\\n").filter(Boolean).map(JSON.parse);
      }} catch (e) {{ alert("Could not parse " + f.name + " as JSONL: " + e.message); return; }}
      if (kind === "resp") {{
        loadedResp = rows;
        alert(`Loaded ${{rows.length}} responses from ${{f.name}}.\\nNow load the matching scores.jsonl.`);
        return;
      }}
      if (!loadedResp) {{ alert("Load responses.jsonl first."); return; }}
      const sc = Object.fromEntries(rows.map(o => [o.response_id, o]));
      const steer = loadedResp[0].coef !== undefined;
      const built = loadedResp.map(r => {{
        const s = sc[r.response_id] || {{}};
        const base = {{
          p: r.persona, q: r.question_idx, s: r.sample_idx, r: r.response,
          run: r.run_id || f.name, t: s.trait_score, c: s.coherence_score,
          qt: Q.push(r.question) - 1, sp: SP.push(r.system_prompt || "") - 1,
        }};
        // alpha needs ||v_raw|| from run_meta.json, which a JSONL load does not carry
        return steer ? {{...base, sw: `${{r.persona}} · ${{r.run_id || f.name}}`, coef: r.coef, a: r.coef}}
                     : {{...base, pol: r.polarity, sl: r.split, i: r.instruction_idx}};
      }});
      const missing = built.filter(r => r.t == null).length;
      if (missing) alert(`${{missing}} of ${{built.length}} responses had no matching score — `
                       + `check that the scores file belongs to the same run.`);
      if (steer) {{ ST = built; mode = "st"; }} else {{ EV = built; mode = "ev"; }}
      $("mode").value = mode;
      selP = new Set(); selC = new Set(); loadedResp = null;
      setup();
    }};
    rd.readAsText(f);
    fp.value = "";
  }};
  fp.click();
}}

["samp", "instr", "split", "npp", "start"].forEach(id => $(id).addEventListener("input", render));
$("sweep").addEventListener("change", syncSweep);
$("mode").addEventListener("change", e => {{ mode = e.target.value; setup(); }});
setup();
</script></body></html>""", encoding="utf-8")
print(f"-> {OUT}  ({OUT.stat().st_size:,} bytes)  eval rows {len(ev)}, steer rows {len(st)}")
