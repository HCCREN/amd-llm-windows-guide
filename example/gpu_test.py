import torch
import time

device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
dtype = torch.float16

def test_linear(size):
    print(f"\n=== Linear Test: {size} x {size} ===")
    layer = torch.nn.Linear(size, size, dtype=dtype).to(device)
    x = torch.randn(size, size, dtype=dtype, device=device)

    torch.cuda.synchronize()
    start = time.time()
    y = layer(x)
    torch.cuda.synchronize()

    print("Forward time:", time.time() - start)

if __name__ == "__main__":
    for size in [1024, 2048, 4096, 8192]:
        test_linear(size)

