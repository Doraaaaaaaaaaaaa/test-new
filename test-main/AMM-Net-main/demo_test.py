import os
import argparse
import torch
from transformers import BertTokenizer, BertModel

from models import catNet
from losses import emd_loss, binary_accuracy, cal_metrics
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_LEN = 200

_imagenet_norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    _imagenet_norm,
])


def load_sample(img_path, text, tokenizer, score_label=None):
    img = Image.open(img_path).convert("RGB")
    image = _transform(img).unsqueeze(0)

    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    text_ids = enc["input_ids"]        # (1, MAX_LEN)
    text_mask = enc["attention_mask"]  # (1, MAX_LEN)

    label = None
    if score_label is not None:
        label = torch.tensor(score_label, dtype=torch.float32)
        label = label / (label.sum() + 1e-8)
        label = label.unsqueeze(0)

    return image, text_ids, text_mask, label


def run(model, tokenizer, device, img_path, text, score_label=None):
    model.eval()

    image, text_ids, text_mask, label = load_sample(
        img_path, text, tokenizer, score_label
    )
    image     = image.to(device)
    text_ids  = text_ids.to(device)
    text_mask = text_mask.to(device)
    if label is not None:
        label = label.to(device)

    with torch.no_grad():
        output = model(image, text_ids, text_mask)

    bins = torch.arange(1, 11, dtype=torch.float32, device=device)
    pred_mean = (output * bins).sum(dim=1).item()

    print("Predicted distribution:")
    print(output.squeeze(0).cpu().numpy())
    print(f"Predicted mean score: {pred_mean:.4f}")

    if label is not None:
        criterion = emd_loss(dist_r=1)
        loss_val = criterion(output, label).item()
        acc_val  = binary_accuracy(output, label, bins=10).item()
        metrics  = cal_metrics(
            [output.cpu().numpy()],
            [label.cpu().numpy()],
            bins=10,
        )
        print("GT distribution:")
        print(label.squeeze(0).cpu().numpy())
        print(f"EMD loss:       {loss_val:.4f}")
        print(f"Binary accuracy:{acc_val:.4f}")
        print(f"Pred mean / GT mean / acc(%): "
              f"{metrics[0][0]:.4f} / {metrics[1][0]:.4f} / {metrics[2]:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      type=str,   required=True,
                        help="训练保存的模型权重路径")
    parser.add_argument("--image",           type=str,   required=True,
                        help="测试图片路径")
    parser.add_argument("--text",            type=str,   required=True,
                        help="对应文本描述")
    parser.add_argument("--label",           type=float, nargs=10, default=None,
                        help="可选：10维真实分布标签，用于计算 EMD / accuracy")
    parser.add_argument("--parn_pretrained", type=str,   default="",
                        help="AMM-Net.pt 或 PARN 预训练权重路径（与训练时保持一致）")
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert      = BertModel.from_pretrained("bert-base-uncased")
    model     = catNet(
        bert,
        parn_pretrained_path=args.parn_pretrained if args.parn_pretrained else None,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")

    run(
        model=model,
        tokenizer=tokenizer,
        device=device,
        img_path=args.image,
        text=args.text,
        score_label=args.label,
    )


if __name__ == "__main__":
    main()
