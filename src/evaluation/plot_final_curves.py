import matplotlib.pyplot as plt
import numpy as np

def plot_final_results():
    epochs = np.arange(1, 21)
    val_accs = [
        86.70, 92.12, 92.30, 92.53, 93.87, 92.71, 93.35, 93.64, 94.69, 93.12,
        92.94, 93.93, 92.47, 95.45, 93.76, 92.94, 95.39, 93.93, 94.81, 94.52
    ]
    peak_epoch = np.argmax(val_accs) + 1
    peak_acc = val_accs[peak_epoch - 1]

    train_loss_epochs = np.arange(1, 21)
    train_losses = [
        0.2105, 0.4810, 0.2380, 0.0032, 0.0053, 0.0116, 0.1639, 0.4366, 0.0218, 0.0008,
        0.0061, 0.0008, 0.0008, 0.0002, 0.0022, 0.0574, 0.0007, 0.0002, 0.0026, 0.0002
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    plt.style.use('seaborn-v0_8-paper')

    ax1.plot(train_loss_epochs, train_losses, 'r-o', markersize=4, label='Training Loss (Last Step in Epoch)')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Final Model (CNN-LSTM) Training & Validation History', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.plot(epochs, val_accs, 'b-s', markersize=4, label='Jester Validation Accuracy')
    
    ax2.scatter(peak_epoch, peak_acc, s=150, c='red', marker='*', zorder=5)
    ax2.annotate(f'Peak: {peak_acc:.2f}% (Model Saved)', 
                 xy=(peak_epoch, peak_acc), xytext=(peak_epoch-4, peak_acc-3),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                 fontsize=11, fontweight='bold', color='red')
    
    ax2.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(80, 100) 
    ax2.legend()

    plt.tight_layout()
    filename = 'learning_curves_final.png'
    plt.savefig(filename, dpi=300) 
    print(f"Successfully generated Final Learning Curves: {filename}")
    plt.show()

if __name__ == "__main__":
    plot_final_results()