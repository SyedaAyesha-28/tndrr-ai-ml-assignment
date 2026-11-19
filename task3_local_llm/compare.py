import csv
import sys

# Usage: python compare.py              (reads results.csv)
#        python compare.py results_cpu.csv
filename = sys.argv[1] if len(sys.argv) > 1 else "results.csv"

by_prompt = {}
with open(filename, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        by_prompt.setdefault(row["prompt"], {})[row["model"]] = row["output"]

fp16_name = next(m for p in by_prompt.values() for m in p if "fp16" in m)
q4_name = next(m for p in by_prompt.values() for m in p if "q4" in m)

# Also write a scoring sheet you can open in Excel and fill in
with open("quality_scores.csv", "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(["#", "prompt", "fp16_output", "q4_output",
                "fp16_score_1to5", "q4_score_1to5", "notes"])
    for i, (prompt, answers) in enumerate(by_prompt.items(), start=1):
        w.writerow([i, prompt, answers[fp16_name], answers[q4_name], "", "", ""])

        print("=" * 80)
        print(f"PROMPT {i}: {prompt}\n")
        print(f"--- fp16 ---\n{answers[fp16_name].strip()}\n")
        print(f"--- q4_K_M ---\n{answers[q4_name].strip()}\n")

print("Saved quality_scores.csv (fill in the score and notes columns).")