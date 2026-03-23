"""
预计算 CLIP 图像特征。

用法:
  python precompute_clip.py \
    --csv /path/to/train.csv \
    --images_dir /path/to/images \
    --output_dir /root/autodl-tmp/clip_features \
    --batch_size 64

训练时使用:
  python train_ava.py \
    --images_dir /path/to/images \
    --train_csv /path/to/train.csv \
    --val_csv /path/to/val.csv \
    --precompute_clip /root/autodl-tmp/clip_features
"""
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms
import pandas as pd
import clip

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageOnlyDataset(Dataset):
    def __init__(self, csv_path, images_dir):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir

        clip_norm = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            clip_norm,
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = str(self.df.iloc[idx]["image_id"])
        img_path = os.path.join(self.images_dir, image_id)
        if not os.path.exists(img_path):
            if not image_id.lower().endswith(".jpg") and os.path.exists(img_path + ".jpg"):
                img_path = img_path + ".jpg"
            else:
                raise FileNotFoundError(img_path)

        img = Image.open(img_path).convert("RGB")
        return self.transform(img), image_id


def all_candidate_keys(image_id: str):
    image_id = str(image_id)
    keys = [image_id]

    base, ext = os.path.splitext(image_id)
    if ext:
        keys.append(base)
    else:
        keys.append(image_id + ".jpg")
        keys.append(image_id + ".png")

    return list(dict.fromkeys(keys))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_name", type=str, default="ViT-B/16")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = clip.load(args.clip_name, device=device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    ds = ImageOnlyDataset(args.csv, args.images_dir)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    all_features = {}

    with torch.no_grad():
        for i, (imgs, ids) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            feats = model.encode_image(imgs)
            feats = F.normalize(feats.float(), dim=-1)

            for j, img_id in enumerate(ids):
                feat = feats[j].cpu()
                for k in all_candidate_keys(img_id):
                    all_features[k] = feat

            if (i + 1) % 100 == 0:
                print(f"Processed {(i + 1) * args.batch_size} images...")

    save_path = os.path.join(args.output_dir, "clip_features.pt")
    torch.save(all_features, save_path)
    print(f"Saved {len(all_features)} CLIP features to {save_path}")


if __name__ == "__main__":
    main()