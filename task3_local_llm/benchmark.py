import requests, csv

MODELS = ["qwen2.5:0.5b-instruct-fp16", "qwen2.5:0.5b-instruct-q4_K_M"]

PROMPTS = [
    # Factual (2)
    "What is the capital of Australia, and what is one common misconception about it?",
    "Explain in two sentences what photosynthesis does for a plant.",

    # Reasoning / math (2)
    "A train travels 120 km in 1.5 hours. At the same speed, how long will it take to travel 200 km? Show your steps.",
    "I have 3 red balls and 5 blue balls in a bag. If I pick two without replacement, what is the probability both are red? Show your working.",

    # Code (2)
    "Write a Python function that checks whether a string is a palindrome, ignoring spaces and capitalisation.",
    "Write a Python function that returns the second largest number in a list without using sort().",

    # Instruction-following with format constraints (2)
    "List exactly 3 benefits of regular exercise as 3 bullet points, each under 10 words.",
    "Describe the moon in exactly two sentences. Do not use the letter 'z'.",

    # Summarization / rewriting (2)
    "Summarize this in one sentence: 'Remote work has grown rapidly in recent years. Companies report savings on office space, while employees enjoy flexible schedules. However, some workers feel isolated and find it hard to separate work from home life.'",
    "Rewrite this more politely: 'Send me the report now, I can't keep waiting for you.'",
]


def run(model, prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_predict": 512},
    }).json()
    tps = r["eval_count"] / (r["eval_duration"] / 1e9)
    prompt_tps = r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9)
    return r["response"], tps, prompt_tps


rows = []
for m in MODELS:
    print(f"Running {m} ...")
    run(m, "warm up")  # first call includes model load time, so we discard it
    for p in PROMPTS:
        out, tps, ptps = run(m, p)
        rows.append([m, p, out, round(tps, 2), round(ptps, 2)])
        print(f"  done: {p[:50]}... ({round(tps, 2)} tok/s)")

with open("results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "prompt", "output", "gen_tok_s", "prompt_tok_s"])
    writer.writerows(rows)

print("Done. Saved results.csv")
