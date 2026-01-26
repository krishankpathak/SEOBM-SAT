# src/analysis/plot_learning_curves.py
# ======================================
# Plot learning curves for Baseline vs SEOBM
# ======================================

import matplotlib.pyplot as plt

# Hard-code logged values (from your run)
epochs = [0, 10, 20, 30, 40]
baseline = [0.1122, 0.0700, 0.0435, 0.0301, 0.0244]
seobm = [0.0644, 0.0567, 0.0523, 0.0461, 0.0357]

plt.figure(figsize=(6,4))
plt.plot(epochs, baseline, marker='o', label="Baseline (no memory)")
plt.plot(epochs, seobm, marker='s', label="SEOBM (temporal)")

plt.xlabel("Training Epoch")
plt.ylabel("MSE (normalized)")
plt.title("Delayed Prediction Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
