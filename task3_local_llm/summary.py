import csv
import sys
from collections import defaultdict

# Usage: python summary.py            (reads results.csv)
#        python summary.py results_cpu.csv
filename = sys.argv[1] if len(sys.argv) > 1 else "results.csv"

gen = defaultdict(list)
prompt = defaultdict(list)

with open(filename, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        gen[row["model"]].append(float(row["gen_tok_s"]))
        prompt[row["model"]].append(float(row["prompt_tok_s"]))

print(f"Results from {filename}\n")
for model in gen:
    g, p = gen[model], prompt[model]
    print(model)
    print(f"  generation: mean {sum(g)/len(g):.1f} tok/s  (min {min(g):.1f}, max {max(g):.1f})")
    print(f"  prompt processing: mean {sum(p)/len(p):.1f} tok/s")
    print()