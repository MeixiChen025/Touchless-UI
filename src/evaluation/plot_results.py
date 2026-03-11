import matplotlib.pyplot as plt

epochs_baseline = [1, 2, 3, 4, 5]
acc_baseline = [23.69, 28.24, 34.19, 35.53, 34.19]

epochs_primary = list(range(1, 21))
acc_primary = [
    57.70, 64.94, 69.25, 69.78, 66.98,
    70.42, 69.54, 70.89, 70.95, 70.25,
    70.83, 68.38, 68.84, 68.79, 69.37,
    70.07, 68.26, 69.78, 68.73, 68.90
]

plt.figure(figsize=(10, 6))

plt.plot(epochs_baseline, acc_baseline, marker='o', linestyle='--', color='gray', label='Baseline (2D CNN)')

plt.plot(epochs_primary, acc_primary, marker='s', linestyle='-', color='blue', label='Primary (3D CNN)')

plt.title('Validation Accuracy: Baseline vs. Primary Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.xticks(range(1, 21))
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()