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

# ── Transforms ────────────────────────────────────────────────────────────────
# Training: geometric-only augmentation (no ColorJitter) to stay consistent
# with the PARN offline cache, which scores the original image colours.
def _make_transform(size=448, is_train=False):
    if is_train:
        ops = [
            transforms.Resize(size),
            transforms.RandomCrop(size, padding=0),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        ops = [
            transforms.Resize(size),
            transforms.CenterCrop(size),
        ]
    ops += [transforms.ToTensor(), _imagenet_norm]
    return transforms.Compose(ops)


class AVACaptionsDataset(Dataset):
    """
    CSV required columns:
      image_id, comment, and one of:
      - prob_1..prob_10
      - score1..score10
      - score2..score11

    Args:
        parn_cache_path: optional path to a .pt file produced by
            precompute_parn.py, containing {image_id: {'g': Tensor(2048),
            'scores': Tensor(11)}}. When provided, __getitem__ returns
            (image, text_ids, text_mask, y, parn_g, parn_scores) and the
            dataset always returns 6 items (zeros when a key is missing).

    __getitem__ returns:
        (image, text_ids, text_mask, y, parn_g, parn_scores)
        parn_g     : (2048,) float — zeros when no cache
        parn_scores: (11,)   float — zeros when no cache
    """

    _ZERO_G      = torch.zeros(2048)
    _ZERO_SCORES = torch.zeros(11)

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        tokenizer,
        max_len: int = 128,
        is_train: bool = False,
        parn_cache_path: str = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = _make_transform(size=448, is_train=is_train)

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
        self.all_input_ids      = enc_all["input_ids"]
        self.all_attention_mask = enc_all["attention_mask"]

        # precompute normalised labels
        scores_np = self.df[self.score_cols].values.astype("float32")
        scores_sum = scores_np.sum(axis=1, keepdims=True) + 1e-8
        self.all_labels = torch.from_numpy(scores_np / scores_sum)

        # image_id list for cache lookup
        self.image_ids = self.df["image_id"].astype(str).tolist()

        # PARN feature cache
        self.parn_cache = None
        if parn_cache_path and os.path.exists(parn_cache_path):
            print(f"Loading PARN cache from {parn_cache_path} ...")
            self.parn_cache = torch.load(
                parn_cache_path, map_location="cpu", weights_only=False
            )
            print(f"  {len(self.parn_cache)} entries loaded.")
            # Coverage check: warn if cache misses are significant
            unique_ids = set(self.image_ids)
            missing = [
                img_id for img_id in unique_ids
                if img_id not in self.parn_cache
                and img_id.replace(".jpg", "") not in self.parn_cache
                and img_id + ".jpg" not in self.parn_cache
            ]
            hit_rate = 1.0 - len(missing) / max(1, len(unique_ids))
            print(f"  Cache hit rate: {hit_rate*100:.1f}%  "
                  f"({len(unique_ids) - len(missing)}/{len(unique_ids)} images)")
            if missing:
                print(f"  WARNING: {len(missing)} image IDs not in cache "
                      f"— they will use zero attribute features.")
                if len(missing) <= 20:
                    for m in missing:
                        print(f"    missing: {m}")

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

    def _get_parn(self, image_id: str):
        if self.parn_cache is None:
            return self._ZERO_G, self._ZERO_SCORES
        entry = self.parn_cache.get(image_id) or \
                self.parn_cache.get(image_id.replace(".jpg", "")) or \
                self.parn_cache.get(image_id + ".jpg")
        if entry is None:
            return self._ZERO_G, self._ZERO_SCORES
        return entry["g"], entry["scores"]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, image_id)
        if not os.path.exists(img_path):
            if not image_id.lower().endswith(".jpg") and \
               os.path.exists(img_path + ".jpg"):
                img_path = img_path + ".jpg"
            else:
                raise FileNotFoundError(img_path)

        img   = Image.open(img_path).convert("RGB")
        image = self.transform(img)

        text_ids  = self.all_input_ids[idx]
        text_mask = self.all_attention_mask[idx]
        y         = self.all_labels[idx]
        parn_g, parn_scores = self._get_parn(image_id)

        return image, text_ids, text_mask, y, parn_g, parn_scores
