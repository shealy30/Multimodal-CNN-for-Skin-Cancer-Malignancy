# # Ensemble dataset (Use this part before training metadata model)
# import pandas as pd
# from PIL import Image
# from torch.utils.data import Dataset

# class HAMDataset(Dataset):
#     def __init__(self, csv_path, transform=None, metadata_transform=None):
#         self.df = pd.read_csv(csv_path)
#         self.transform = transform
#         self.metadata_transform = metadata_transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         image = Image.open(row["image_path"]).convert("RGB")
#         label = int(row["target"])

#         metadata = {
#             "age": row["age"],
#             "sex": row["sex"],
#             "localization": row["localization"],
#             "dx_type": row["dx_type"]
#         }

#         if self.transform:
#             image = self.transform(image)

#         return image, metadata, label


## Image only dataset
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HAMDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["target"])

        if self.transform:
            image = self.transform(image)

        return image, label