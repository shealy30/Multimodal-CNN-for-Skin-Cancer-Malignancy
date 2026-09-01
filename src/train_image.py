import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.dataset import HAMDataset
from src.config import (
    PROCESSED_DIR, OUTPUT_DIR, BATCH_SIZE, NUM_WORKERS,
    EPOCHS, LEARNING_RATE, IMG_SIZE
)
from src.plot_utils import (
    save_training_curves,
    save_confusion_matrix,
    save_selected_predictions
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, return_samples=False):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    sample_images = []

    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().ravel()
        preds = (probs >= 0.5).astype(int)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

        if return_samples:
            for img in images.cpu():
                sample_images.append(img)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    if return_samples:
        return acc, f1, y_true, y_pred, y_prob, sample_images
    return acc, f1, y_true, y_pred


def main():
    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    fig_dir = OUTPUT_DIR / "figures" / "image_model"
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_ds = HAMDataset(PROCESSED_DIR / "train.csv", transform=train_transform)
    val_ds = HAMDataset(PROCESSED_DIR / "val.csv", transform=eval_transform)
    test_ds = HAMDataset(PROCESSED_DIR / "test.csv", transform=eval_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    n_pos = (train_df["target"] == 1).sum()
    n_neg = (train_df["target"] == 0).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)

    print(f"Using pos_weight = {pos_weight.item():.4f}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1.0
    best_model_path = OUTPUT_DIR / "models" / "best_image_model.pt"

    train_losses = []
    val_accs = []
    val_f1s = []

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc, val_f1, val_true, val_pred, val_prob, _ = evaluate(
            model, val_loader, return_samples=True
        )

        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)

        val_df = pd.DataFrame({
            "y_true": val_true,
            "y_pred": val_pred,
            "y_prob": val_prob
        })

        val_df.to_csv(fig_dir / "val_predictions.csv", index=False)

    print(f"\nBest validation F1: {best_val_f1:.4f}")
    print(f"Saved best model to: {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_acc, test_f1, y_true, y_pred, y_prob, sample_images = evaluate(
        model, test_loader, return_samples=True
    )
    # Save predictions to CSV
    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob
    })

    pred_path = fig_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print(f"\nSaved prediction outputs to: {pred_path}")

    save_training_curves(
        train_losses,
        val_accs,
        val_f1s,
        str(fig_dir / "image_training")
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        str(fig_dir / "image_confusion_matrix.png")
    )

    save_selected_predictions(
        sample_images,
        y_true,
        y_pred,
        y_prob,
        str(fig_dir / "correct_predictions.png"),
        correct=True,
        max_images=8
    )

    save_selected_predictions(
        sample_images,
        y_true,
        y_pred,
        y_prob,
        str(fig_dir / "incorrect_predictions.png"),
        correct=False,
        max_images=8
    )

    print(f"\nSaved figures to: {fig_dir}")


if __name__ == "__main__":
    main()