🚀 AMD LLM Windows Guide
Run Qwen / DeepSeek / Llama on AMD Radeon RX 9070 XT (Pure Windows, No Linux Required)










This guide demonstrates exactly how to run modern LLM models (Qwen, DeepSeek, Llama, TinyLlama) on an AMD Radeon RX 9070 XT GPU using pure Windows —

❌ No Linux
❌ No WSL
❌ No dual-boot

Everything here is 100% tested on real hardware and fully reproducible.

📚 Table of Contents

What This Project Covers

Tested Environment

1. Install ROCm HIP SDK

2. Create Python Virtual Environment

3. Install ROCm PyTorch

4. Verify GPU Access

5. Run Your First LLM

6. Stress Test (Linear / Transformer)

Future Work

🔧 What This Project Covers

Install ROCm HIP SDK for Windows

Install ROCm-enabled PyTorch

Verify GPU support (hipinfo, torch.cuda)

Stress-test Linear & Transformer layers

Run HuggingFace LLMs (TinyLlama, Qwen, DeepSeek)

Prepare for RAG troubleshooting assistant (coming soon)

🖥️ Tested Environment
Component	Version
OS	Windows 11 Pro 24H2
GPU	AMD Radeon RX 9070 XT
Driver	AMD ROCm HIP SDK 6.4.2
Python	3.12.10
PyTorch	ROCm nightly build

⭐ 1. Install ROCm HIP SDK (Windows)

Download:
🔗 https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html

Choose:

Windows 10 & 11

ROCm 6.4.2 HIP SDK

After installation, check device support:
hipinfo
You should see:
device 0: AMD Radeon(TM) Graphics
device 1: AMD Radeon RX 9070 XT

⭐ 2. Create Python Virtual Environment

python -m venv venv
venv\Scripts\activate

⭐ 3. Install ROCm PyTorch

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

⭐ 4. Verify GPU Access

import torch

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"Device {i}:", torch.cuda.get_device_name(i))

Expected output:
CUDA available: True
Device 0: AMD Radeon(TM) Graphics
Device 1: AMD Radeon RX 9070 XT

⭐ 5. Run Your First LLM on AMD GPU

Example: TinyLlama (Fast & Lightweight)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
dtype = torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map={"": device},
)

prompt = "You are a helpful assistant. Briefly introduce yourself."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

⭐ 6. Stress Test Transformer / Linear on AMD GPU

import torch
import time

device = torch.device("cuda:1")
dtype = torch.float16

for size in [1024, 2048, 4096, 8192]:
    print(f"\n=== Testing Linear Layer: {size} x {size} ===")
    layer = torch.nn.Linear(size, size, dtype=dtype).to(device)

    x = torch.randn(size, size, dtype=dtype, device=device)

    start = time.time()
    y = layer(x)
    torch.cuda.synchronize()
    print("Forward time:", time.time() - start)

🔮 Future Work (RAG / Qwen / DeepSeek)

Add Qwen2.5-7B-Instruct full GPU guide

Add DeepSeek local inference

Add RAG pipeline for troubleshooting logs

Add stress test suite for diagnosing ROCm instability

❤️ Contributions

PRs & suggestions are welcome — especially around:

ROCm tuning

AMD GPU optimization

Windows + PyTorch stability tips

Model benchmarks

📄 License

MIT License.



















