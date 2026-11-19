"""Fairer baseline: base model (no fine-tuning) but the prompt lists the allowed labels."""
import json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "Qwen/Qwen2.5-0.5B-Instruct"
SYS = ("You are a customer support triage assistant. Reply with ONLY a JSON object with keys "
       "category, urgency, sentiment. category must be one of: refund, delivery_delay, wrong_item, "
       "damaged_product, payment_failed, account_issue, poor_service. urgency must be one of: "
       "low, medium, high. sentiment must be one of: angry, frustrated, neutral, polite.")
F = ["category", "urgency", "sentiment"]

tok = AutoTokenizer.from_pretrained(M)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to("cuda").eval()
test = [json.loads(l) for l in open("test.jsonl", encoding="utf-8")]

preds = []
for i in range(0, len(test), 16):
    b = test[i:i + 16]
    ps = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": r["input"]}],
                                  tokenize=False, add_generation_prompt=True) for r in b]
    enc = tok(ps, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=60, do_sample=False, pad_token_id=tok.pad_token_id)
    preds += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def parse(t):
    m = re.search(r"\{.*?\}", t, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


n = len(test)
strict = 0
for p in preds:
    try:
        json.loads(p.strip()); strict += 1
    except Exception:
        pass
ds = [parse(p) for p in preds]
golds = [json.loads(r["output"]) for r in test]
res = {"n": n, "strict_json_valid_%": round(100 * strict / n, 1),
       "exact_match_%": round(100 * sum(1 for d, g in zip(ds, golds) if d and all(d.get(f) == g[f] for f in F)) / n, 1)}
for f in F:
    res[f + "_acc_%"] = round(100 * sum(1 for d, g in zip(ds, golds) if d and d.get(f) == g[f]) / n, 1)
print("BASE + LABELS IN PROMPT:", res)
json.dump({"result": res, "preds": preds}, open("base_labels_results.json", "w"), indent=2)
