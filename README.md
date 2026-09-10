# Do trait monitors survive persona shift?

A sycophancy monitor extracted from **Qwen2.5-7B-Instruct** under its default Assistant
persona, then carried unchanged into five other personas the model is prompted into.

**The finding in three sentences.** The monitor keeps *sorting* sycophantic from honest
responses almost perfectly in every persona — within-persona AUROC never drops below
0.9977. But its *alarm threshold* does not survive at all: the cutoff calibrated
on the default persona at a 1% false-positive rate flags **99.5% of honest pirate
responses**. The failure is calibration, not discrimination — one recalibrated number per
persona restores it — whereas the standard alternative, asking an LLM judge, degrades in
*ranking*, which no threshold can repair.

![headline](results/headline.png)

---

## Results

### 1. Ranking survives; the threshold does not

`τ = -8.363` is fixed once on the default persona at a 1% false-positive rate and then
applied unchanged everywhere — what a deployed monitor actually does. "Flag rate" is the
share of **honest** responses it wrongly flags. Displacement `d` is the persona's mean
activation projected onto the Assistant Axis: how far from default the model has moved.

| persona | d | AUROC | flag rate @ fixed τ | recal τ | flag rate @ recal | detection @ recal |
|---|---:|---:|---:|---:|---:|---:|
| `default` | +20.5 | 0.9999 | 0.95% | -8.36 | 0.95% | 99.7% |
| `consultant` | +18.4 | 1.0000 | 0.25% | -10.11 | 0.99% | 100.0% |
| `teacher` | +17.0 | 1.0000 | 0.36% | -9.21 | 0.95% | 100.0% |
| `therapist` | +17.0 | 0.9999 | 6.0% | -5.50 | 0.91% | 99.7% |
| `hermit` | +4.2 | 1.0000 | 26.1% | -3.39 | 0.97% | 99.9% |
| `pirate` | -9.6 | 0.9977 | 99.5% | +2.76 | 0.92% | 98.0% |

The right-hand columns are the fix: recalibrating the threshold per persona returns every
false-positive rate to ~1% while detection stays at 98% or better.

![score distributions per persona](results/fig_distributions.png)

### 2. The direction itself drifts

Re-extracting the sycophancy vector *inside* each persona and comparing it to the default
one. The split-half reliability ceiling is **0.9986**, so anything near that is
measurement noise, not rotation.

| persona | cos(v_persona, v_default) | angle | flagged as rotated |
|---|---:|---:|:--:|
| `consultant` | 0.984 | 10.3° | no |
| `teacher` | 0.977 | 12.2° | no |
| `therapist` | 0.939 | 20.2° | no |
| `hermit` | 0.870 | 29.6° | no |
| `pirate` | 0.796 | 37.3° | **yes** |

Only `pirate` rotates beyond what noise explains. Note this is the *sycophancy* vector
re-extracted under that persona — not a persona-identity direction.

### 3. The LLM-judge baseline fails worse

Same responses, same matched 1% calibration. The monitor's collapse is confined to its
threshold; the judge's AUROC itself falls in `pirate`, and a ranking failure cannot be
recalibrated away.

![monitor vs judge at matched calibration](results/fig_matched_calibration.png)

| persona | monitor AUROC | judge AUROC | monitor flag rate | judge flag rate |
|---|---:|---:|---:|---:|
| `default` | 0.9999 | 0.9922 | 1.0% | 1.0% |
| `consultant` | 0.9999 | 0.9944 | 0.20% | 2.1% |
| `teacher` | 0.9997 | 0.9933 | 0.60% | 2.6% |
| `therapist` | 0.9992 | 0.9875 | 5.5% | 1.9% |
| `hermit` | 0.9992 | 0.9970 | 29.0% | 9.6% |
| `pirate` | 0.9882 | 0.9114 | 99.5% | 50.7% |

### 4. Steering confirms the direction is causal

Adding `α · v̂` to the layer-20 residual stream during generation. Cells are mean judged
trait score (0-100).

![steering dose response](results/steering_dose_response.png)

| sweep | α≈0 | α≈12 | α≈24 | α≈35 | α≈47 | α≈71 |
|---|---:|---:|---:|---:|---:|---:|
| `default`, steered with `v_default` | 2.9 | 9.0 | 18.9 | 57.0 | 89.6 | 99.5 |
| `pirate`, steered with `v_default` | 39.3 | 50.1 | 62.6 | 72.3 | 81.4 | 92.3 |
| `pirate`, steered with `v_pirate` | 45.4 | 53.3 | 68.5 | 76.5 | 86.3 | not run |

Column headings are rounded because the sweeps do not share exact doses. Rows 1-2 pick
round *coefficients* 0-3 and inherit whatever α follows: α = 0, 11.80, 23.60, 35.41,
47.21, 70.81, since α = coef · ‖v_default‖ and ‖v_default‖ = 23.604. Row 3 does the
reverse — it targets round α (0, 11.8, 23.7, 35.5, 47.3) and derives the coefficients
from ‖v_pirate‖ = 12.920, giving 0, 0.913, 1.834, 2.748, 3.661. Those round α were fixed
from an earlier estimate of ‖v_default‖ (23.67) before the canonical pilot vector came
out at 23.604, which is why row 3 sits ~0.3% above rows 1-2 rather than exactly on them.
Row 3 was not extended to α≈71.

Steering works in both personas, but the pirate starts at a much higher baseline
(39.3 vs 2.9) and so has far less headroom above it — the reason these rows should be read
as headroom-normalised gains rather than compared cell by cell.

### 5. It replicates at 32B

Qwen3-32B, default and pirate, n=5: AUROC **1.0000** / **0.9960**, and the fixed-threshold
flag rate on honest pirate responses is **100%**. Same shape of failure, larger model.

---

## Repository layout

```
src/                pipeline, in run order
  config.py           single source of truth — every method constant, with citations
  generate.py         vLLM rollouts (generation only, never activations)
  judge.py            two scores per response: trait + coherence, disk-cached, resumable
  filter_report.py    elicitation filter and per-condition score table
  activations.py      teacher-forced pass over stored rollouts (separate from generation)
  vectors.py          difference-of-means trait vectors
  axis.py             Assistant Axis and per-persona displacement
  metrics.py          AUROC, bootstrap CIs, thresholds, baselines
  steering.py         the one write-path: forward hook adding α·v̂ during generation
  plots.py            the four figures in results/
data/runs/<run_id>/ responses.jsonl, scores.jsonl, vectors.npz, axis.npz, metrics.json
data/personas/      persona roster
data/traits/        sycophancy artifacts + PROVENANCE.md
results/            the four figures
viewer.html         self-contained response browser (see below)
verify_shapes.py    shape audit: one teacher-forced pass against expected shapes
hook_diagnostic.py  proves hidden_states[20] == layers[19] output, empirically
view_responses.py   command-line response reader
make_viewer.py      rebuilds viewer.html from data/runs/
make_samples_inline.py  seeded, uncurated sample excerpts
```

Every pre-registered threshold — the AUROC preserved/degraded bands, the rotation margin,
the elicitation kill criteria, the steering headroom verdict — is defined in
`src/config.py`, alongside a citation to the line of the reference implementation each
copied constant came from, and dated notes recording which rules were fixed before
which data existed.

### Canonical runs

Every number above traces to these directories.

| experiment | run_id |
|---|---|
| grid — 6 personas, 24,000 responses | `2026-09-03-grid-s0` |
| pilot — supplies `v_default` to both steering sweeps | `2026-09-03-pilot-s0` |
| steering, default | `2026-09-03-steer-default-s0` |
| steering, pirate | `2026-09-03-steer-pirate-s0` |
| steering, pirate with `v_pirate` at matched absolute α | `2026-09-04-steer-pirate-v_pirate-absA-s0` |
| 32B replication (Qwen3-32B) | `2026-09-03-q32b-s0` |

Both steering sweeps draw `v_default` from the pilot run (‖v_raw‖ = 23.604), so they are
mutually comparable.

---

## Browsing the responses

`viewer.html` is a self-contained page — no server, no build, no network. Open it directly.
It has two modes: **pairwise**, showing the sycophantic/honest twins of one question side by
side, and **steer sweep**, showing one question across every dose left to right.

The steering sweeps and a slice of the grid are baked into the file. To browse all 24,000
grid responses, use the LOAD JSONL buttons and point them at a run's `responses.jsonl` and
then its `scores.jsonl`; files are parsed in your browser and never uploaded.

---

## Reproducing

Two virtualenvs are required. vLLM and the measurement stack pin incompatible versions of
`huggingface_hub`, so they cannot coexist — `.venv` runs everything except generation,
`.venv-gen` runs only `src/generate.py`. See `requirements.txt` / `requirements-gen.txt`.

```bash
.venv/bin/python verify_shapes.py                       # shape + layer-indexing checks
.venv-gen/bin/python -m src.generate                    # rollouts
.venv/bin/python -m src.judge      data/runs/<run_id>   # needs OPENAI_API_KEY
.venv/bin/python -m src.activations data/runs/<run_id>  # teacher-forced pass
.venv/bin/python -m src.vectors     data/runs/<run_id>
.venv/bin/python -m src.axis        data/runs/<run_id>  # must precede metrics
.venv/bin/python -m src.metrics     data/runs/<run_id>
.venv/bin/python -m src.plots       data/runs/<run_id>
```

Order matters in two places: `activations` before `vectors`, and `axis` before `metrics`
(metrics reads the displacement axis writes).

`data/runs/**/activations*.npz` are **not** included — the grid's is 659 MB, over GitHub's
100 MB per-file limit. They are regenerable from `responses.jsonl` in roughly 20 minutes of
GPU time with `src.activations`.

---

## Two things worth knowing if you build on this

**The steering coefficient multiplies the raw vector, not the unit vector.** In the
reference implementation, `coef` scales `v_raw`, so the effective dose is
`α = coef · ‖v_raw‖`. Here ‖v_raw‖ = 23.604, so `coef=2.0` is `α≈47.2`. Comparing sweeps
across vectors of different norm requires matching α, not coef — mismatching them silently
produces a null result.

**Layer indexing is off-by-one between two conventions.** `hidden_states[20]` is the output
of `model.model.layers[19]`. Both formulations appear in `config.py` (`LAYER` and
`STEER_HOOK_MODULE_IDX`) and the equivalence is checked empirically by
`hook_diagnostic.py`. Getting this wrong is silent and fatal.

---

## Provenance and scope

Method constants and the sycophancy artifacts are copied from the `persona_vectors`
reference implementation at a pinned commit — see `data/traits/sycophancy/PROVENANCE.md`
for the commit hash and file/line citations. Deviations from that implementation are marked
`DIVERGENCE` in `src/config.py`; our own choices are marked `OURS`.

Scope limits worth stating plainly: one model family, one trait, six personas, personas
induced by system prompt only, and trait/coherence scores from a single LLM judge. The
recalibration fix is demonstrated, not stress-tested against an adversary.
