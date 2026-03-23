import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_transforms():
    imagenet_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform_main = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        imagenet_norm,
    ])

    clip_norm = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    transform_clip = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        clip_norm,
    ])
    return transform_main, transform_clip


class AVACaptionsDataset(Dataset):
    """
    CSV required columns:
      image_id, comment, and one of:
      - prob_1..prob_10
      - score1..score10
      - score2..score11
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        tokenizer,
        max_len: int = 200,
        clip_feature_path: str = "",
    ):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform_main, self.transform_clip = build_transforms()

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

        # precompute normalized labels
        scores_np = self.df[self.score_cols].values.astype("float32")
        scores_sum = scores_np.sum(axis=1, keepdims=True) + 1e-8
        self.all_labels = torch.from_numpy(scores_np / scores_sum)

        # optional precomputed CLIP features
        self.use_precomputed_clip = False
        self.clip_features = None
        if clip_feature_path:
            self.clip_features = self._load_clip_feature_file(clip_feature_path)
            self.use_precomputed_clip = True
            print(f"Loaded {len(self.clip_features)} precomputed CLIP features from {clip_feature_path}")

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

    @staticmethod
    def _load_clip_feature_file(path: str):
        """
        path can be:
        - directory containing clip_features.pt
        - direct path to clip_features.pt
        """
        if os.path.isdir(path):
            path = os.path.join(path, "clip_features.pt")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Precomputed CLIP feature file not found: {path}")

        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError(f"Expected dict in {path}, got {type(obj)}")
        return obj

    @staticmethod
    def _candidate_feature_keys(image_id: str):
        image_id = str(image_id)
        keys = [image_id]

        base, ext = os.path.splitext(image_id)
        if ext:
            keys.append(base)
        else:
            keys.append(image_id + ".jpg")
            keys.append(image_id + ".png")

        return list(dict.fromkeys(keys))

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
        image = self.transform_main(img)

        if self.use_precomputed_clip:
            feat = None
            for k in self._candidate_feature_keys(image_id):
                if k in self.clip_features:
                    feat = self.clip_features[k]
                    break
            if feat is None:
                raise KeyError(
                    f"Cannot find precomputed CLIP feature for image_id='{image_id}'. "
                    f"Tried keys: {self._candidate_feature_keys(image_id)}"
                )

            if not isinstance(feat, torch.Tensor):
                feat = torch.tensor(feat)
            image_att_or_feat = feat.float()
        else:
            image_att_or_feat = self.transform_clip(img)

        text_ids = self.all_input_ids[idx]
        text_mask = self.all_attention_mask[idx]
        y = self.all_labels[idx]

        return image, text_ids, text_mask, image_att_or_feat, y