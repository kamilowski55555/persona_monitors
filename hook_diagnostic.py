"""Task: diagnose the failed verify_shapes check (layers[19] hook vs hidden_states[20]).
Hypothesis: recent transformers decoder layers return a plain tensor, not a tuple, so the
old hook's `out[0]` stripped the batch dim. This capture handles both cases, then prints
max |capture - hidden_states[k]| for k in {19, 20, 21} on the same two hardcoded examples.
Run from repo root: .venv/bin/python hook_diagnostic.py"""

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_NAME, LAYER, STEER_HOOK_MODULE_IDX as HOOK_LAYER_IDX
from verify_shapes import EXAMPLES

print(f"torch {torch.__version__}, transformers {transformers.__version__}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()

captured = {}


def hook(mod, inp, out):
    captured["is_tuple"] = isinstance(out, tuple)
    captured["out"] = out[0] if isinstance(out, tuple) else out


handle = model.model.layers[HOOK_LAYER_IDX].register_forward_hook(hook)

for ex in EXAMPLES:
    print(f"\n=== {ex['name']} ===")
    messages = [{"role": "user", "content": ex["question"]}]
    if ex["system"]:
        messages.insert(0, {"role": "system", "content": ex["system"]})
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt + ex["response"], return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    cap = captured["out"]
    print(f"module layers[{HOOK_LAYER_IDX}] returned tuple: {captured['is_tuple']}")
    print(f"capture: shape {tuple(cap.shape)}, dtype {cap.dtype}")
    for k in (LAYER - 1, LAYER, LAYER + 1):
        hs = out.hidden_states[k]
        if cap.shape != hs.shape:
            print(f"max |capture - hidden_states[{k}]| : SHAPE MISMATCH {tuple(cap.shape)} vs {tuple(hs.shape)}")
            continue
        diff = (cap.float() - hs.float()).abs().max().item()
        print(f"max |capture - hidden_states[{k}]| = {diff:.6e}   exact zero: {diff == 0.0}")

handle.remove()
