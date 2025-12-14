# Examples

This directory contains runnable scripts for AMD LLM Windows Guide.

## Files

### `check_env.py`
Verifies:
- Python version  
- PyTorch build  
- ROCm availability  
- GPU names  
- hipinfo output  

### `gpu_test.py`
Stress tests:
- Linear layers  
- Transformer-like matrix multiplies  
Useful for detecting ROCm stability issues.

### `run_qwen.py`
Runs Qwen2.5-7B-Instruct on AMD GPU.

---

## Running Examples

Activate your virtual environment:

```bat
venv\Scripts\activate
