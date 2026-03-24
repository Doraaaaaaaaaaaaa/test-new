"""
Precompute frozen PARN backbone features (g, scores) for all images.

Saves: {image_id: {"g": Tensor(2048), "scores": Tensor(11)}}

Usage:
    python precompute_parn.py \
        --images_dir   C:/Users/admin/AVA_Dataset/images \
        --csv          C:/Users/admin/AVA_Dataset/train.csv \
        --csv          C:/Users/admin/AVA_Dataset/val.csv \
        --parn_weights C:/Users/admin/Desktop/test-main/test-main/AMM-Net-main/AMM-Net.pt \
        --output       C:/Users/admin/AVA_Dataset/parn_cache.pt \
        --batch_size   64
"""

import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _norm,
])


# ── Minimal PARN backbone (matches AMM-Net.pt img_attr.* keys) ────────────────
class _PARNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet       = tv_models.resnet50(weights=None)
        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc1_1   = nn.Linear(2048, 256)
        self.bn1_1   = nn.BatchNorm1d(256)
        self.relu1_1 = nn.PReLU()
        self.fc2_1   = nn.Linear(256, 64)
        self.relu2_1 = nn.PReLU()
        self.fc3_1   = nn.Linear(64, 11)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        g = x.flatten(1)                              # (B, 2048)
        h = self.relu1_1(self.bn1_1(self.fc1_1(g)))  # (B, 256)
        h = self.relu2_1(self.fc2_1(h))               # (B, 64)
        scores = self.fc3_1(h)                         # (B, 11)
        return g, scores


class _ImageDataset(Dataset):
    def __init__(self, image_ids, images_dir):
        self.image_ids  = image_ids
        self.images_dir = images_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, img_id)
        if not os.path.exists(img_path):
            alt = img_path + ".jpg"
            img_path = alt if os.path.exists(alt) else img_path
        try:
            img = Image.open(img_path).convert("RGB")
            return _transform(img), img_id, False   # (tensor, id, is_bad)
        except Exception:
            return torch.zeros(3, 224, 224), img_id, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir",   type=str, required=True)
    parser.add_argument("--csv",          type=str, nargs="+", required=True,
                        help="One or more CSV files (train / val / test)")
    parser.add_argument("--parn_weights", type=str, required=True,
                        help="AMM-Net.pt containing img_attr.* weights")
    parser.add_argument("--output",       type=str, required=True,
                        help="Output .pt file path")
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--num_workers",  type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load backbone
    backbone = _PARNBackbone().to(device)
    ckpt = torch.load(args.parn_weights, map_location="cpu", weights_only=False)
    if any(k.startswith("img_attr.") for k in ckpt):
        state = {k[len("img_attr."):]: v for k, v in ckpt.items()
                 if k.startswith("img_attr.")}
        missing, _ = backbone.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: missing keys in backbone: {missing[:5]}")
    else:
        backbone.load_state_dict(ckpt, strict=False)
    backbone.eval()
    print("PARN backbone loaded.")

    # Collect unique image IDs across all CSVs
    all_ids = set()
    for csv_path in args.csv:
        df = pd.read_csv(csv_path)
        all_ids.update(df["image_id"].astype(str).tolist())
    image_ids = sorted(all_ids)
    print(f"Total unique images: {len(image_ids)}")

    ds     = _ImageDataset(image_ids, args.images_dir)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    cache = {}
    bad_images = []
    with torch.no_grad():
        for i, (imgs, ids, bad_flags) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            g_batch, scores_batch = backbone(imgs)
            g_batch      = g_batch.cpu()
            scores_batch = scores_batch.cpu()
            for j, img_id in enumerate(ids):
                if bad_flags[j].item():
                    bad_images.append(img_id)
                    continue          # skip bad images — do NOT write to cache
                cache[img_id] = {
                    "g":      g_batch[j],
                    "scores": scores_batch[j],
                }
            if (i + 1) % 100 == 0:
                done = min((i + 1) * args.batch_size, len(image_ids))
                print(f"  {done}/{len(image_ids)} images processed")

    if bad_images:
        print(f"\nWARNING: {len(bad_images)} images failed to load and were skipped:")
        for p in bad_images[:20]:
            print(f"  {p}")
        if len(bad_images) > 20:
            print(f"  ... and {len(bad_images) - 20} more")
        bad_path = os.path.splitext(args.output)[0] + "_bad_images.txt"
        with open(bad_path, "w") as f:
            f.write("\n".join(bad_images))
        print(f"Full bad-image list saved → {bad_path}")
    else:
        print("All images loaded successfully.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(cache, args.output)
    print(f"Saved {len(cache)} entries → {args.output}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
