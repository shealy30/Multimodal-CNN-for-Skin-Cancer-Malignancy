
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


BASE_PATH = r"C:\Users\anasir5\Courses\MultiModalML\skin_cancer_multimodal\data\raw"

metadata_path = os.path.join(BASE_PATH, "HAM10000_metadata.csv")
img_path_1 = os.path.join(BASE_PATH, "HAM10000_images_part_1")
img_path_2 = os.path.join(BASE_PATH, "HAM10000_images_part_2")

OUT_DIR = r"C:\Users\anasir5\Courses\MultiModalML\skin_cancer_multimodal\outputs\figures\eda"
os.makedirs(OUT_DIR, exist_ok=True)


df = pd.read_csv(metadata_path)

print("Data loaded.")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


MALIGNANT = {"mel", "bcc", "akiec"}
BENIGN = {"nv", "bkl", "df", "vasc"}

def make_binary_label(dx):
    if dx in MALIGNANT:
        return "Malignant"
    elif dx in BENIGN:
        return "Benign"
    return np.nan

df["binary_label"] = df["dx"].apply(make_binary_label)

# keep only rows used in your project
df = df[df["binary_label"].notna()].copy()


def get_image_path(image_id):
    p1 = os.path.join(img_path_1, image_id + ".jpg")
    p2 = os.path.join(img_path_2, image_id + ".jpg")
    if os.path.exists(p1):
        return p1
    elif os.path.exists(p2):
        return p2
    return None

df["image_path"] = df["image_id"].apply(get_image_path)
df = df[df["image_path"].notna()].copy()


print("\n========== DATASET SUMMARY ==========")
print(f"Total samples used: {len(df)}")
print("\nBinary label counts:")
print(df["binary_label"].value_counts())

print("\nBinary label percentages:")
print((df["binary_label"].value_counts(normalize=True) * 100).round(2))

print("\nMetadata variables used:")
print(["age", "sex", "localization", "dx_type"])


sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120


plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x="binary_label", order=["Benign", "Malignant"])
plt.title("Binary Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "binary_class_distribution.png"))
plt.show()

class_pct = df["binary_label"].value_counts(normalize=True).reindex(["Benign", "Malignant"]) * 100

plt.figure(figsize=(6, 4))
bars = plt.bar(class_pct.index, class_pct.values)
plt.title("Class Proportions (%)")
plt.ylabel("Percentage")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.1f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "binary_class_percentages.png"))
plt.show()


plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="binary_label", y="age", order=["Benign", "Malignant"])
plt.title("Age Distribution by Class")
plt.xlabel("Class")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "age_by_class.png"))
plt.show()


plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="binary_label", hue="sex", order=["Benign", "Malignant"])
plt.title("Sex Distribution by Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sex_by_class.png"))
plt.show()

top_locs = df["localization"].value_counts().head(8).index
df_loc = df[df["localization"].isin(top_locs)].copy()

plt.figure(figsize=(10, 5))
sns.countplot(
    data=df_loc,
    x="localization",
    hue="binary_label",
    order=top_locs
)
plt.title("Top Lesion Locations by Class")
plt.xlabel("Localization")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "location_by_class.png"))
plt.show()


sample_df = (
    df.groupby("binary_label", group_keys=False)
      .apply(lambda x: x.sample(4, random_state=42))
      .reset_index(drop=True)
)

plt.figure(figsize=(12, 6))

for i in range(len(sample_df)):
    row = sample_df.iloc[i]
    img = Image.open(row["image_path"]).convert("RGB")

    plt.subplot(2, 4, i + 1)
    plt.imshow(img)
    plt.title(f"{row['binary_label']}\n{row['dx']}")
    plt.axis("off")

plt.suptitle("Sample Dermoscopic Images")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sample_images.png"))
plt.show()


print("\n========== PROJECT MODALITIES ==========")
print("Image modality: dermoscopic lesion image")
print("Metadata modality: age, sex, localization, dx_type")

print("\nEDA complete. Figures saved to:")
print(OUT_DIR)  