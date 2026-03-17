import matplotlib.pyplot as plt


def plot_loss_curves(histories):
    plt.figure(figsize=(8, 5))
    for h in histories:
        label = f"{h['optimizer'].upper()} (lr={h['learning_rate']})"
        plt.plot(h["epoch"], h["train_loss"], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Loss Convergence Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_accuracy_curves(histories):
    plt.figure(figsize=(8, 5))
    for h in histories:
        label = f"{h['optimizer'].upper()} (lr={h['learning_rate']})"
        plt.plot(h["epoch"], h["test_acc"], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()