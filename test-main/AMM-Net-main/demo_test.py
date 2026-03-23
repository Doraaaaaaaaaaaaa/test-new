import os
import argparse
import torch
from transformers import BertModel

from models import catNet
from losses import emd_loss, binary_accuracy, cal_metrics
from preprocess import txt_process, transform_test
from PIL import Image


def load_demo_sample(img_path, txt, score_label=None):
    img = Image.open(img_path).convert("RGB")

    image = transform_test(img).unsqueeze(0)
    txt_ids = txt_process(txt).unsqueeze(0)
    text_mask = (txt_ids != 0).long()

    if score_label is None:
        label = None
    else:
        label = torch.tensor(score_label, dtype=torch.float32)
        label = label / (label.sum() + 1e-8)
        label = label.unsqueeze(0)

    return image, txt_ids, text_mask, label


def test(model, device, img_path, txt, score_label=None):
    model.eval()

    image, txt_ids, text_mask, label = load_demo_sample(img_path, txt, score_label)

    image = image.to(device)
    txt_ids = txt_ids.to(device)
    text_mask = text_mask.to(device)

    if label is not None:
        label = label.to(device)

    criterion_aes_val = emd_loss(dist_r=1)

    with torch.no_grad():
        output = model(image, txt_ids, text_mask)

    print("Predicted distribution:")
    print(output.squeeze(0).cpu().numpy())

    pred_mean = torch.sum(output * torch.arange(1, 11, device=device).float(), dim=1)
    print("Predicted mean score:", pred_mean.item())

    if label is not None:
        loss_val = criterion_aes_val(output, label).item()
        acc_val = binary_accuracy(output, label, bins=10).item()

        metrics = cal_metrics(
            [output.cpu().numpy()],
            [label.cpu().numpy()],
            bins=10,
        )

        print("GT distribution:")
        print(label.squeeze(0).cpu().numpy())
        print("Validation EMD:", loss_val)
        print("Binary accuracy:", acc_val)
        print("Pred mean / GT mean / cls acc(%):", metrics[0][0], metrics[1][0], metrics[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="训练得到的模型权重")
    parser.add_argument("--image", type=str, required=True, help="测试图片路径")
    parser.add_argument("--text", type=str, required=True, help="对应文本描述")
    parser.add_argument(
        "--label",
        type=float,
        nargs=10,
        default=None,
        help="可选：10维真实分布标签，用于计算 EMD / accuracy",
    )
    parser.add_argument(
        "--parn_pretrained",
        type=str,
        default="",
        help="PARN属性编码器预训练权重路径（与训练时保持一致）",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    bert = BertModel.from_pretrained("bert-base-uncased")
    model = catNet(
        bert,
        parn_pretrained_path=args.parn_pretrained if args.parn_pretrained else None,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    test(
        model=model,
        device=device,
        img_path=args.image,
        txt=args.text,
        score_label=args.label,
    )


if __name__ == "__main__":
    main()
