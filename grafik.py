import matplotlib.pyplot as plt
import numpy as np

sizes = ['50×50', '100×100', '500×500', '1000×1000']
time_serial = [0.0012739, 0.00904, 1.27878, 12.7672]
time_omp = [0.0015322, 0.0417872, 0.272, 1.63557]
time_mpi = [0.00000632, 0.00008612, 0.016237, 0.575421]
time_cuda = [0.00000641, 0.00008713, 0.018536, 0.693423]

plt.figure(figsize=(12, 7))

plt.plot(sizes, time_serial, 'o-', color='#1f77b4', linewidth=2.5, 
         markersize=10, label='Последовательная', alpha=0.9)
plt.plot(sizes, time_omp, 's-', color='#ff7f0e', linewidth=2.5, 
         markersize=10, label='OpenMP', alpha=0.9)
plt.plot(sizes, time_mpi, 'D-', color='#2ca02c', linewidth=2.5, 
         markersize=10, label='MPI (12 процессов)', alpha=0.9)
plt.plot(sizes, time_cuda, '^-', color='#9467bd', linewidth=2.5, 
         markersize=10, label='CUDA', alpha=0.9)

for i in range(len(sizes)):
    offset = max(time_serial)*0.02
    plt.text(i, time_serial[i]+offset, f'{time_serial[i]:.5f}', 
             ha='center', color='#1f77b4', fontsize=9)
    plt.text(i, time_omp[i]+offset, f'{time_omp[i]:.5f}', 
             ha='center', color='#ff7f0e', fontsize=9)
    plt.text(i, time_mpi[i]-offset*3, f'{time_mpi[i]:.2e}', 
             ha='center', color='#2ca02c', fontsize=9)
    plt.text(i, time_cuda[i]-offset*3, f'{time_cuda[i]:.2e}', 
             ha='center', color='#9467bd', fontsize=9)

plt.ylim(0, max(time_serial)*1.15) 
plt.ylabel('Время выполнения (секунды)', fontsize=12)
plt.xlabel('Размер матрицы', fontsize=12)
plt.title('Сравнение производительности умножения матриц', fontsize=14, pad=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11, framealpha=1)

plt.tight_layout()
plt.savefig('matrix_multiplication_linear_comparison.png', dpi=300, bbox_inches='tight')
plt.show()