import matplotlib.pyplot as plt
import os

def plot_training_curve(values_dict, title, out_path):
    """
    values_dict: {"label1": list, "label2": list}
    """
    plt.figure(figsize=(7,5))
    for label, values in values_dict.items():
        plt.plot(values, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[Plot saved] {out_path}")
