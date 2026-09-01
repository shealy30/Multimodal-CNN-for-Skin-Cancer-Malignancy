import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.utils import transform_metadata_row


class HAMMultimodalDataset(Dataset):
    def __init__(self, csv_path, scaler, encoder, transform=None):
        self.df = pd.read_csv(csv_path)
        self.scaler = scaler
        self.encoder = encoder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        metadata = transform_metadata_row(row, self.scaler, self.encoder)
        metadata = torch.tensor(metadata, dtype=torch.float32)

        label = torch.tensor(int(row["target"]), dtype=torch.float32)

        return image, metadata, label