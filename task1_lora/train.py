"""
Task 1: LoRA fine-tune Qwen2.5-0.5B-Instruct for complaint triage -> JSON.
Needs train.jsonl in the same folder (run make_data.py first).
Run:  python train.py        (or paste into a notebook cell)
"""
import json
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = ("You are a customer support triage assistant. Read the complaint and reply "
          "with ONLY a JSON object with keys category, urgency, sentiment.")

# ---- LoRA settings (you must be able to explain each one) ----
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LR = 2e-4
EPOCHS = 2
BATCH = 8
GRAD_ACCUM = 2
MAX_LEN = 256

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model = get_peft_model(model, LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=TARGETS, task_type="CAUSAL_LM"))
model.print_trainable_parameters()  # note these numbers for your report


def prompt_text(complaint):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": complaint}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode(row):
    p = tok(prompt_text(row["input"]), add_special_tokens=False)["input_ids"]
    a = tok(row["output"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p + a)[:MAX_LEN]
    labels = ([-100] * len(p) + a)[:MAX_LEN]  # loss only on the answer, not the prompt
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


rows = [json.loads(l) for l in open("train.jsonl", encoding="utf-8")]
ds = Dataset.from_list(rows).map(encode, remove_columns=["input", "output"])

args = TrainingArguments(
    output_dir="lora_tmp", num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR, lr_scheduler_type="cosine", warmup_steps=10,
    fp16=True, logging_steps=10, save_strategy="no", report_to="none")

trainer = Trainer(model=model, args=args, train_dataset=ds,
                  data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
trainer.train()

model.save_pretrained("lora_out")
tok.save_pretrained("lora_out")
print("Saved adapter to lora_out/")
