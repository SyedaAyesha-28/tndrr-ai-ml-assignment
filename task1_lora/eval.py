"""
Task 1 evaluation: base model vs LoRA fine-tuned model on test.jsonl.
Run after train.py:  python eval.py
Saves results.json (all predictions) for your failure analysis.
"""
import json
import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = ("You are a customer support triage assistant. Read the complaint and reply "
          "with ONLY a JSON object with keys category, urgency, sentiment.")
FIELDS = ["category", "urgency", "sentiment"]

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
test = [json.loads(l) for l in open("test.jsonl", encoding="utf-8")]


def generate_all(model, batch_size=16):
    outs = []
    for i in range(0, len(test), batch_size):
        batch = test[i:i + batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["input"]}],
            tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=60, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def parse_strict(text):
    try:
        d = json.loads(text.strip())
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def parse_lenient(text):  # finds the first {...} even if the model added extra words
    m = re.search(r"\{.*?\}", text, re.S)
    return parse_strict(m.group(0)) if m else None


def score(preds):
    n = len(test)
    strict = lenient = exact = 0
    field_ok = {f: 0 for f in FIELDS}
    for r, p in zip(test, preds):
        gold = json.loads(r["output"])
        if parse_strict(p) is not None:
            strict += 1
        d = parse_lenient(p)
        if d is not None:
            lenient += 1
            for f in FIELDS:
                if d.get(f) == gold[f]:
                    field_ok[f] += 1
            if all(d.get(f) == gold[f] for f in FIELDS):
                exact += 1
    res = {"n": n,
           "strict_json_valid_%": round(100 * strict / n, 1),
           "lenient_json_valid_%": round(100 * lenient / n, 1),
           "exact_match_%": round(100 * exact / n, 1)}
    for f in FIELDS:
        res[f"{f}_acc_%"] = round(100 * field_ok[f] / n, 1)
    return res


base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda")
base.eval()
base_preds = generate_all(base)
base_res = score(base_preds)
print("BASE      :", base_res)

ft = PeftModel.from_pretrained(base, "lora_out")
ft.eval()
ft_preds = generate_all(ft)
ft_res = score(ft_preds)
print("FINE-TUNED:", ft_res)

json.dump({"base": base_res, "finetuned": ft_res,
           "rows": [{"input": r["input"], "gold": r["output"], "base": b, "ft": f}
                    for r, b, f in zip(test, base_preds, ft_preds)]},
          open("results.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# print failures of the fine-tuned model (pick 3 for your report)
print("\nFINE-TUNED FAILURES:")
shown = 0
for r, f in zip(test, ft_preds):
    d = parse_lenient(f)
    if d is None or any(d.get(k) != json.loads(r["output"])[k] for k in FIELDS):
        print("-", r["input"], "\n   gold:", r["output"], "\n   pred:", f.strip())
        shown += 1
        if shown >= 10:
            break
