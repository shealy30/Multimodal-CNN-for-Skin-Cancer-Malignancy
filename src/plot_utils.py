import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_training_curves(train_losses, val_accs, val_f1s, save_path_prefix):
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_accs, marker="o", label="Validation Accuracy")
    plt.plot(epochs, val_f1s, marker="s", label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_metrics.png", dpi=300)
    plt.close()


def save_confusion_matrix(y_true, y_pred, save_path, class_names=("Benign", "Malignant")):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def denormalize_image(img_tensor):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img_tensor.cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))


def save_selected_predictions(images, true_labels, pred_labels, probs, save_path, correct=True, max_images=8):
    selected = []
    for img, t, p, prob in zip(images, true_labels, pred_labels, probs):
        is_correct = (t == p)
        if is_correct == correct:
            selected.append((img, t, p, prob))
        if len(selected) == max_images:
            break

    if len(selected) == 0:
        return

    cols = 4
    rows = int(np.ceil(len(selected) / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, (img, t, p, prob) in enumerate(selected):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(denormalize_image(img))
        plt.axis("off")
        plt.title(
            f"True: {'Mal' if t == 1 else 'Ben'}\n"
            f"Pred: {'Mal' if p == 1 else 'Ben'} ({prob:.2f})"
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()