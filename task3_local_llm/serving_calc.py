# Back-of-envelope memory calculator for serving a model to many users.
# Change the numbers in the "ASSUMPTIONS" block, then run: python serving_calc.py

# ---------- MODEL (Llama 3 8B, from its config.json) ----------
params_billion = 8        # number of parameters, in billions
layers = 32
kv_heads = 8
head_dim = 128

# ---------- ASSUMPTIONS (you choose these, and state them in your report) ----------
bits_per_weight = 4.5     # 4-bit weights plus a little extra for scales
kv_bytes = 2              # bytes per KV value (2 = fp16)
context_tokens = 2048     # average context length per user
users = 20
overhead_gb = 2           # CUDA context, activations, etc.
gpu_gb = 24

# ---------- CALCULATION ----------
weights_gb = params_billion * 1e9 * (bits_per_weight / 8) / 1e9

kv_per_token = 2 * layers * kv_heads * head_dim * kv_bytes   # bytes
kv_per_user_gb = kv_per_token * context_tokens / 1e9
kv_total_gb = kv_per_user_gb * users

total_gb = weights_gb + kv_total_gb + overhead_gb

print(f"Weights:              {weights_gb:.2f} GB")
print(f"KV cache per token:   {kv_per_token:,} bytes")
print(f"KV cache per user:    {kv_per_user_gb:.3f} GB  (at {context_tokens} tokens)")
print(f"KV cache, {users} users:   {kv_total_gb:.2f} GB")
print(f"Overhead (assumed):   {overhead_gb:.2f} GB")
print(f"TOTAL:                {total_gb:.2f} GB  of {gpu_gb} GB")
print("Fits!" if total_gb <= gpu_gb else "Does NOT fit.")