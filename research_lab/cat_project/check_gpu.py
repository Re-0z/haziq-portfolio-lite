import torch
import time

print(f"PyTorch version: {torch.__version__}")
print("----------------------------------")

if torch.cuda.is_available():
    print("CUDA is available!")
    print(f" GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f" VRAM Status: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print("\n[Running Speed Test...]")

    size = 1000

    start = time.time()
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)
    z_cpu = torch.matmul(x_cpu, y_cpu)
    end = time.time()
    print(f" CPU Time: {end - start:.4f} seconds")

    start = time.time()
    x_gpu = torch.randn(size, size).to('cuda')
    y_gpu = torch.randn(size, size).to('cuda')
    torch.cuda.synchronize()
    end = time.time()
    print(f" GPU Time: {end - start:.4f} seconds")

else:
    print(" Error: CUDA not detected. Running on CPU only.")