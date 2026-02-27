import torch
import time

def test_speed(dtype, size=4096):
    a = torch.randn(size, size, device='cuda', dtype=dtype)
    b = torch.randn(size, size, device='cuda', dtype=dtype)
    
    # Warmup
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print(f"Dtype: {dtype}, Time: {time.time() - start:.4f}s")

if __name__ == "__main__":
    try:
        test_speed(torch.float32)
        test_speed(torch.float16)
        test_speed(torch.bfloat16)
    except Exception as e:
        print(e)
