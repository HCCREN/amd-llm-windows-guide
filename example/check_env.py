import torch

import subprocess

print("=== Python & PyTorch Environment ===")

print("Python version:", subprocess.getoutput("python --version"))

print("PyTorch version:", torch.__version__)

print("CUDA available:", torch.cuda.is_available())

print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
  
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

print("\n=== HIP Devices (hipinfo) ===")

try:
  
    print(subprocess.getoutput("hipinfo"))
  
except:
  
    print("hipinfo not found. Make sure HIP SDK is installed.")

