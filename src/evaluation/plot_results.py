import matplotlib.pyplot as plt

epochs_baseline = [1, 2, 3, 4, 5]
acc_baseline = [23.98, 25.85, 28.12, 30.16, 31.91]

epochs_primary = [1, 2, 3]
acc_primary = [60.97, 63.36, 65.93]

plt.figure(figsize=(8, 5))

plt.plot(epochs_baseline, acc_baseline, marker='o', linestyle='--', color='gray', label='Baseline (2D CNN)')

plt.plot(epochs_primary, acc_primary, marker='s', linestyle='-', color='blue', label='Primary (3D CNN)')

plt.title('Validation Accuracy: Baseline vs. Primary Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.xticks([1, 2, 3, 4, 5])
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()