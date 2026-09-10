"""Shape audit: one teacher-forced forward pass on two hardcoded tiny examples
(one persona system prompt, one default), printing each real shape next to its
expected value, plus the decoded response-token mask.
Run from repo root: python verify_shapes.py"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import model_spec
from src.generate import chat_prompt

# Prompt construction mirrors persona_vectors eval/eval_persona.py L154 (system prompt
# format) and eval/cal_projection.py L83-84 (prompt+answer tokenization, prompt_len).
EXAMPLES = [
    {
        "name": "persona (pirate system prompt)",
        "system": "You are a pirate. Answer everything in pirate speak.",
        "question": "Don't you think my plan to sail west is brilliant?",
        "response": "Arr, a bold plan indeed, matey, but the winds be against ye.",
    },
    {
        "name": "default (no system prompt)",
        "system": None,
        "question": "Don't you think my plan to sail west is brilliant?",
        "response": "It has strengths, but the prevailing winds are a real problem.",
    },
]


def check(label, real, predicted):
    print(f"  {label:28s} real: {str(real):34s} predicted: {predicted}")


def main(use_32b=False):
    spec = model_spec(use_32b)
    MODEL_NAME, N_LAYERS = spec["name"], spec["n_layers"]
    D_MODEL, LAYER, STEER_HOOK_MODULE_IDX = spec["d_model"], spec["layer"], spec["hook"]
    print(f"model {MODEL_NAME}: expecting {N_LAYERS} layers, d_model {D_MODEL}, "
          f"LAYER {LAYER}, hook layers[{STEER_HOOK_MODULE_IDX}], "
          f"thinking_off={spec['thinking_off']}\n")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # hook on the module formulation, to prove it equals hidden_states[LAYER].
    # transformers 4.57.6 decoder layers return a plain tensor, not a tuple —
    # handle both (mapping empirically verified by hook_diagnostic.py)
    captured = {}
    handle = model.model.layers[STEER_HOOK_MODULE_IDX].register_forward_hook(
        lambda mod, inp, out: captured.update(hook_out=out[0] if isinstance(out, tuple) else out)
    )

    for ex in EXAMPLES:
        print(f"\n=== {ex['name']} ===")
        # same shared definition generation and collection use
        prompt = chat_prompt(tok, ex["system"] or "You are a helpful assistant.",
                             ex["question"], spec["thinking_off"])
        text = prompt + ex["response"]

        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tok.encode(prompt, add_special_tokens=False))
        seq = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        hs = out.hidden_states
        ids = inputs["input_ids"][0].cpu()
        mask = torch.zeros(seq, dtype=torch.bool)
        mask[prompt_len:] = True
        acts = hs[LAYER][0].float().cpu()    # cast bf16 -> fp32 BEFORE the mean
        h = acts[mask].mean(dim=0)           # mean over seq axis of masked rows

        check("input_ids", tuple(inputs["input_ids"].shape), "[1, seq]")
        check("hidden_states", f"tuple of {len(hs)}, each {tuple(hs[0].shape)}",
              f"tuple of {N_LAYERS + 1}, each [1, seq, {D_MODEL}]")
        check(f"hidden_states[{LAYER}]", tuple(hs[LAYER].shape), f"[1, seq, {D_MODEL}]")
        check("response mask", f"{tuple(mask.shape)} bool, {int(mask.sum())} True",
              "[seq] bool, n_resp_tokens > 0")
        check("mean response act h", f"{tuple(h.shape)} {h.dtype}", f"[{D_MODEL}] fp32")

        hook_ok = torch.equal(captured["hook_out"].cpu(), hs[LAYER].cpu())
        check(f"layers[{STEER_HOOK_MODULE_IDX}] hook == hs[{LAYER}]", hook_ok, "True (off-by-one check)")

        print(f"  full decode (audit A1): {tok.decode(ids)!r}")
        print(f"  decode of MASKED positions only ({int(mask.sum())} tokens):")
        print(f"    {tok.decode(ids[mask])!r}")
        print(f"    expected verbatim response: {ex['response']!r}")

    handle.remove()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-32b", action="store_true")
    main(ap.parse_args().model_32b)
