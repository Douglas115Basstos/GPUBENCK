
import time
import statistics
import torch
import torch_directml

# Configuração
N = 4096
REPETICOES = 10
AQUECIMENTO = 3

cpu = torch.device("cpu")
gpu = torch_directml.device()

print("=" * 50)
print("       BENCHMARK CPU VS AMD RX 6600")
print("=" * 50)

print("GPU:", torch_directml.device_name(0))
print("PyTorch:", torch.__version__)

# Matrizes de entrada
a_cpu = torch.randn(N, N, device=cpu)
b_cpu = torch.randn(N, N, device=cpu)

a_gpu = a_cpu.to(gpu)
b_gpu = b_cpu.to(gpu)

def executar(dispositivo, a, b):
    tempos = []

    for _ in range(AQUECIMENTO):
        resultado = torch.matmul(a, b)
        resultado.cpu()

    for _ in range(REPETICOES):
        inicio = time.perf_counter()
        resultado = torch.matmul(a, b)

        # Sincroniza a execução transferindo o resultado para CPU
        resultado.cpu()

        tempos.append(time.perf_counter() - inicio)

    return tempos

print("\nTestando CPU...")
tempos_cpu = executar(cpu, a_cpu, b_cpu)

print("Testando RX 6600...")
tempos_gpu = executar(gpu, a_gpu, b_gpu)

cpu_mediana = statistics.median(tempos_cpu)
gpu_mediana = statistics.median(tempos_gpu)

print("\n" + "=" * 50)
print("                 RESULTADOS")
print("=" * 50)

print(f"CPU - mediana: {cpu_mediana:.4f} segundos")
print(f"GPU - mediana: {gpu_mediana:.4f} segundos")

if gpu_mediana < cpu_mediana:
    ganho = cpu_mediana / gpu_mediana
    print(f"\nGPU mais rápida: {ganho:.2f}x")
else:
    diferenca = gpu_mediana / cpu_mediana
    print(f"\nCPU mais rápida: {diferenca:.2f}x")

print("\nBenchmark concluído!")
