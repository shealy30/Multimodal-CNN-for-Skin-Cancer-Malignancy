import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import METADATA_CSV, IMG_DIR_1, IMG_DIR_2, PROCESSED_DIR, RANDOM_STATE

MALIGNANT = {"mel", "bcc", "akiec"}
BENIGN = {"nv", "bkl", "df", "vasc"}

def get_image_path(image_id: str):
    p1 = IMG_DIR_1 / f"{image_id}.jpg"
    p2 = IMG_DIR_2 / f"{image_id}.jpg"
    if p1.exists():
        return str(p1)
    if p2.exists():
        return str(p2)
    return None

def make_binary_label(dx: str):
    if dx in MALIGNANT:
        return 1
    if dx in BENIGN:
        return 0
    return None

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METADATA_CSV)

    df["image_path"] = df["image_id"].apply(get_image_path)
    df = df[df["image_path"].notna()].copy()

    df["target"] = df["dx"].apply(make_binary_label)
    df = df[df["target"].notna()].copy()

    # keep exactly the columns we need
    keep_cols = [
        "lesion_id", "image_id", "image_path",
        "dx", "dx_type", "age", "sex", "localization", "target"
    ]
    df = df[keep_cols].copy()

    # fill metadata missing values
    df["age"] = df["age"].fillna(df["age"].median())
    df["sex"] = df["sex"].fillna("unknown")
    df["localization"] = df["localization"].fillna("unknown")
    df["dx_type"] = df["dx_type"].fillna("unknown")

    # stratified split
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["target"], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["target"], random_state=RANDOM_STATE
    )

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("Saved splits:")
    print("Train:", train_df.shape)
    print("Val:", val_df.shape)
    print("Test:", test_df.shape)
    print("\nClass balance:")
    print(df["target"].value_counts(normalize=True))

if __name__ == "__main__":
    main()