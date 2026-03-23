import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

_imagenet_norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
_transform_main = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    _imagenet_norm,
])


class AVACaptionsDataset(Dataset):
    """
    CSV required columns:
      image_id, comment, and one of:
      - prob_1..prob_10
      - score1..score10
      - score2..score11

    __getitem__ returns: (image, text_ids, text_mask, y)
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        tokenizer,
        max_len: int = 200,
    ):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = _transform_main

        self.score_cols = self._resolve_score_cols(self.df.columns)

        need = ["image_id", "comment", *self.score_cols]
        for c in need:
            if c not in self.df.columns:
                raise ValueError(f"Missing column '{c}' in {csv_path}")

        # pre-tokenize all text once
        print(f"Pre-tokenizing {len(self.df)} comments from {csv_path} ...")
        comments = self.df["comment"].astype(str).tolist()
        enc_all = tokenizer(
            comments,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.all_input_ids = enc_all["input_ids"]
        self.all_attention_mask = enc_all["attention_mask"]

        # precompute normalised labels
        scores_np = self.df[self.score_cols].values.astype("float32")
        scores_sum = scores_np.sum(axis=1, keepdims=True) + 1e-8
        self.all_labels = torch.from_numpy(scores_np / scores_sum)

        print("Dataset init done.")

    @staticmethod
    def _resolve_score_cols(columns):
        columns = set(columns)
        candidates = [
            [f"prob_{i}" for i in range(1, 11)],
            [f"score{i}" for i in range(1, 11)],
            [f"score{i}" for i in range(2, 12)],
        ]
        for cand in candidates:
            if all(c in columns for c in cand):
                return cand
        raise ValueError(
            "Cannot find valid AVA label columns. "
            "Expected one of: prob_1..prob_10 / score1..score10 / score2..score11"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = str(row["image_id"])
        img_path = os.path.join(self.images_dir, image_id)
        if not os.path.exists(img_path):
            if (not image_id.lower().endswith(".jpg")) and os.path.exists(img_path + ".jpg"):
                img_path = img_path + ".jpg"
            else:
                raise FileNotFoundError(img_path)

        img = Image.open(img_path).convert("RGB")
        image = self.transform(img)

        text_ids = self.all_input_ids[idx]
        text_mask = self.all_attention_mask[idx]
        y = self.all_labels[idx]

        return image, text_ids, text_mask, y
