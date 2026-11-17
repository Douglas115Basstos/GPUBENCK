
import torch
import torch_directml
import time

device = torch_directml.device()

print("GPU:", torch_directml.device_name(0))
print("PyTorch:", torch.__version__)

a = torch.rand(4096, 4096).to(device)
b = torch.rand(4096, 4096).to(device)

# Aquecimento
for _ in range(3):
    c = torch.matmul(a, b)
    c.cpu()

inicio = time.perf_counter()

for _ in range(10):
    c = torch.matmul(a, b)

# Sincronização
resultado = c.cpu()

fim = time.perf_counter()

print("Resultado:", resultado.shape)
print("Execuções: 10")
print(f"Tempo total: {fim - inicio:.3f} segundos")
print("Benchmark concluído!")
