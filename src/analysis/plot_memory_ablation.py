# src/analysis/plot_memory_ablation.py
# ======================================
# Memory window ablation plot
# ======================================

import matplotlib.pyplot as plt

K = [1, 5, 10]
mse = [0.062, 0.044, 0.036]  # representative

plt.figure(figsize=(5,4))
plt.plot(K, mse, marker='o')

plt.xlabel("Memory Window Size (K)")
plt.ylabel("MSE (normalized)")
plt.title("Effect of Memory Window on Prediction Error")
plt.grid(True)
plt.tight_layout()
plt.show()
