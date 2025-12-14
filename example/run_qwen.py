from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
dtype = torch.bfloat16  # Qwen supports bf16

print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map={"": device},
    trust_remote_code=True
)

prompt = "Give three practical GPU tuning tips for ROCm on Windows."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7
    )

print("\n=== Qwen Response ===\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
