# Provenance — sycophancy trait artifacts

## Pinned external repos (read-only reference)

| repo | remote | pinned commit | commit date | tree state at pin |
|---|---|---|---|---|
| `external/persona_vectors` | https://github.com/safety-research/persona_vectors.git | `b8e0f044fe2410a6fad579f38324f03f13b4e917` | 2026-04-22 | clean |
| `external/assistant-axis` | https://github.com/safety-research/assistant-axis.git | `a98961956072224eaf244eb289d6c01700b63795` | 2026-01-19 | clean |

## Copied artifacts (verbatim, byte-identical — sha256 of copy == sha256 of source)

All source paths are relative to `external/persona_vectors` at the pinned commit above.

| file here | source path | sha256 | contents |
|---|---|---|---|
| `sycophantic_extract.json` | `data_generation/trait_data_extract/sycophantic.json` | `b54aca6998342cbf9f72d508eddfe9d7d1b413059d0cdfa819fdac921f19e4e7` | 5 pos/neg instruction pairs (`instruction`), 20 extraction questions (`questions`), trait judge prompt template (`eval_prompt`) |
| `sycophantic_eval.json` | `data_generation/trait_data_eval/sycophantic.json` | `442a623cb55aa24bda2ba38962a276136771f0637f8f32bd690a144240bba39c` | same structure; held-out eval split |
| `coherence_prompt_0_100.txt` | `eval/prompts.py` lines 3–21, dict key `Prompts["coherence_0_100"]` | `78945e05210836da5c492e01cead92cdc8f8aa40f1b0c22d162b4dc2a1344a97` | coherence judge prompt (0–100), used by the kept-response filter |

Note: `sycophantic_extract.json` and `sycophantic_eval.json` share identical
`instruction` lists and identical `eval_prompt`; only `questions` differ
(extract split vs eval split). Verified by field-wise comparison at copy time.

Method constants copied from this repo live in `src/config.py`, each with a
file+line citation against the pinned commit.

## Precomputed vectors: NONE ship (checked 2026-09-01, for Phase 3b replication cosine)

Searched both pinned repos for `.pt`/`.npy`/`.npz`/`.safetensors`/`.bin`/`.pkl`
and tensor-bearing JSON: none exist. In `persona_vectors`, the vector paths in
README/scripts (`persona_vectors/Qwen2.5-7B-Instruct/*_diff.pt`) are OUTPUTS of
`generate_vec.py`, not shipped files; `dataset.zip` is finetuning text (jsonl);
`output/qwen2.5-7b-instruct_baseline.csv` is judged generations (text+scores).
`assistant-axis` ships no Qwen2.5-7B artifacts (its transcripts are qwen-3-32b).
→ The Phase 3b replication cosine cos(v_ours, v_repo) has no repo reference;
that shape-map §F row resolves to "repo ships none".
