# tndrr AI/ML Internship Assignment

Submission by **Syeda Ayesha** | ay3shaa28@gmail.com | SyedaAyesha-28

This repository contains my work for the three assignment tasks. The PDF report with results and written answers is submitted separately through the form (copy: `report.pdf`).

## Repository structure

```
.
├── task1_lora_finetuning/    # LoRA fine-tuning: complaint triage
├── task2_rag/                # RAG from scratch
├── task3_local_inference/    # Local model + quantisation comparison
├── .gitignore
└── README.md
```

## Task 1: LoRA fine-tuning (complaint triage)

- **Base model:** [model name, e.g. Qwen2.5-0.5B-Instruct]
- **Task:** classify customer complaints into a fixed JSON format (category from 7 labels, urgency, sentiment)
- **Dataset:** template-generated; about 40% of test examples use phrasings not seen in training. [train size / test size, at least 50 test]
- **LoRA settings:** rank 16, alpha 32, target modules q, k, v, o, gate, up, down, learning rate 2e-4, 2 epochs (standard starting values, not tuned)
- **Results (held-out test set):**

| Metric | Base model | Fine-tuned |
|--------|-----------|-----------|
| Strict valid JSON | 78% | 100% |
| Exact-match label names | 0% | 74% |

  Urgency and sentiment were perfect after fine-tuning; all remaining errors are in category. Because of the unseen phrasings, 74% is a stress-test figure, not a real-world estimate.
- **Failure analysis:** 3 failing examples (including Hinglish phrasing) are explained in the report, along with fixes (list labels in the prompt, constrained decoding, more varied training data).
- **Memory estimate for a 7B model:** about 115-120 GB (full fine-tuning), 17-21 GB (LoRA), 7-11 GB (QLoRA). Working is in the report.
- **Files:** `[notebook name].ipynb`, `results/`
- **How to run:** open the notebook in Colab/Kaggle with a GPU runtime and run all cells.

## Task 2: RAG from scratch

- **Documents:** [Consumer Protection Act, 2019], [40] pages
- **Chunk sizes compared:** 100 words (266 chunks) and 300 words (89 chunks)
- **Retrieval results (15 self-written questions, each with its source passage):**

| Chunk size | Method | recall@1 | recall@3 | recall@5 |
|-----------|--------|----------|----------|----------|
| 100 words | dense | 27% | 67% | 73% |
| 100 words | BM25 | 33% | 40% | 60% |
| 100 words | hybrid | 33% | 67% | 80% |
| 300 words | dense | 40% | 60% | 67% |
| 300 words | BM25 | 27% | 53% | 60% |
| 300 words | hybrid | 47% | 60% | 80% |

- **Failures analysed:** Q4 (company ignoring a Commission order) and Q15 (shop sharing personal details). Causes and proposed fixes are in the report. I did not implement a re-ranker, so the fixes are predictions, not results.
- **Files:** `[script or notebook name]`, `questions.json` (15 questions with source passages), `results/`
- **How to run:**
  ```bash
  pip install -r requirements.txt
  python [script name].py
  ```
- No LangChain, LlamaIndex or similar frameworks were used.

## Task 3: Local inference and optimisation

- **Model:** Qwen2.5-0.5B-Instruct, fp16 vs q4_K_M
- **Runtime:** Ollama (llama.cpp runner), Windows
- **Hardware:** Intel Core i7-1195G7, 15.8 GB RAM, NVIDIA MX450 (2 GB VRAM)
- **Method:** same 10 prompts, temperature 0, seed 42, max 512 tokens, one warm-up request discarded. Two runs: default placement and a CPU-only control.
- **Results:**

| Run | Model | Generation (tok/s) | Prompt processing (tok/s) |
|-----|-------|-------------------|---------------------------|
| Default | fp16 | 25.0 | 150.6 |
| Default | q4_K_M | 76.1 | 411.2 |
| CPU-only | fp16 | 21.0 | 469.5 |
| CPU-only | q4_K_M | 47.6 | 450.5 |

| Model | File size | Loaded memory | Placement |
|-------|-----------|--------------|-----------|
| fp16 | 994 MB | 1.4 GB | 60% CPU / 40% GPU |
| q4_K_M | 397 MB | 483 MB | 100% GPU |

- **Quality (10 prompts, hand-scored out of 50):** fp16 40, q4_K_M 38. This is a small sample and an indication, not a firm conclusion.
- **Also in the report:** KV cache explanation, why quantisation saves memory, and a plan to serve 20 users on one 24 GB GPU (Llama 3 8B, 4-bit, about 11.9 GB total).
- **Files:** `benchmark.py`, `benchmark_cpu.py`, `check_outputs.py`, `prompts.txt`, `outputs/`
- **How to run:**
  ```bash
  ollama pull qwen2.5:0.5b-instruct-fp16
  ollama pull qwen2.5:0.5b-instruct-q4_K_M
  python benchmark.py
  ```

## Notes

- Model weights, FAISS indexes and other large files are not committed (see `.gitignore`).
- All numbers in the report come from my own runs.
- No API keys are stored in this repository.

## Contact

Syeda Ayesha | ay3shaa28@gmail.com
