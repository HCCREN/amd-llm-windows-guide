import torch
import sys
import os

def print_success(msg):
    print(f"\033[92m[SUCCESS] {msg}\033[0m")

def print_error(msg):
    print(f"\033[91m[ERROR] {msg}\033[0m")

def verify_system():
    print("🔍 Checking Python Environment for AMD RAG...")
    
    # 1. Check Python Version
    print(f"   Python Version: {sys.version.split()[0]}")
    
    # 2. Check PyTorch Installation
    try:
        print(f"   PyTorch Version: {torch.__version__}")
    except ImportError:
        print_error("PyTorch is NOT installed!")
        return

    # 3. Check ROCm / GPU Availability
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print_success(f"ROCm is active! Found {device_count} GPU(s).")
        
        found_high_end_gpu = False
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"   👉 Device {i}: {gpu_name}")
            
            # Simple check for discrete GPU vs Integrated Graphics
            if "RX" in gpu_name or "Radeon" in gpu_name:
                found_high_end_gpu = True

        if not found_high_end_gpu:
            print("\033[93m[WARNING] Only integrated graphics detected? Ensure your RX 9070 XT is recognized.\033[0m")
        
        # 4. VRAM Check
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            print(f"   VRAM: {total_mem / 1024**3:.2f} GB Total | {free_mem / 1024**3:.2f} GB Free")
            
            # Check for Expandable Segments (Critical for Windows)
            alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            if "expandable_segments:True" in alloc_conf:
                print_success("Memory Fragmentation optimization is ENABLED.")
            else:
                print("\033[93m[TIP] Recommend setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for better performance.\033[0m")
                
        except Exception as e:
            print(f"   Could not read VRAM info: {e}")

    else:
        print_error("PyTorch cannot see your GPU. Did you install the CPU version by mistake?")
        print("   Solution: Reinstall using the specific ROCm index-url provided in the guide.")

if __name__ == "__main__":
    verify_system()
