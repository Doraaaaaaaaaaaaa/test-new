import os
import argparse
import torch
from transformers import BertModel

from models import catNet
from losses import emd_loss, binary_accuracy, cal_metrics
from preprocess import txt_process, transform_test, transform_att
from PIL import Image


def load_clip_feature_file(path: str):
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


def candidate_feature_keys(image_id: str):
    image_id = str(image_id)
    keys = [image_id]

    base, ext = os.path.splitext(image_id)
    if ext:
        keys.append(base)
    else:
        keys.append(image_id + ".jpg")
        keys.append(image_id + ".png")

    return list(dict.fromkeys(keys))


def load_demo_sample(img_path, txt, score_label=None, clip_features=None):
    img = Image.open(img_path).convert("RGB")

    image = transform_test(img).unsqueeze(0)
    txt = txt_process(txt).unsqueeze(0)

    # 更严谨的 mask：padding 为 0
    text_mask = (txt != 0).long()

    # 如果提供了预计算 CLIP 特征，就优先使用
    if clip_features is not None:
        image_id = os.path.basename(img_path)
        feat = None
        tried_keys = candidate_feature_keys(image_id)

        for k in tried_keys:
            if k in clip_features:
                feat = clip_features[k]
                break

        if feat is None:
            raise KeyError(
                f"Cannot find precomputed CLIP feature for image '{image_id}'. "
                f"Tried keys: {tried_keys}"
            )

        if not isinstance(feat, torch.Tensor):
            feat = torch.tensor(feat)
        image_att_or_feat = feat.float().unsqueeze(0)
    else:
        image_att_or_feat = transform_att(img).unsqueeze(0)

    if score_label is None:
        label = None
    else:
        label = torch.tensor(score_label, dtype=torch.float32)
        label = label / (label.sum() + 1e-8)
        label = label.unsqueeze(0)

    return image, txt, text_mask, image_att_or_feat, label


def test(model, device, img_path, txt, score_label=None, clip_features=None):
    model.eval()

    image, txt, text_mask, image_att_or_feat, label = load_demo_sample(
        img_path, txt, score_label, clip_features=clip_features
    )

    image = image.to(device)
    txt = txt.to(device)
    text_mask = text_mask.to(device)
    image_att_or_feat = image_att_or_feat.to(device)

    if label is not None:
        label = label.to(device)

    criterion_aes_val = emd_loss(dist_r=1)

    with torch.no_grad():
        output = model(image, txt, text_mask, image_att_or_feat)

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
            bins=10
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
        help="可选：10维真实分布标签，用于计算 EMD / accuracy"
    )
    parser.add_argument(
        "--precompute_clip",
        type=str,
        default="",
        help="预计算CLIP特征缓存目录，或 clip_features.pt 文件路径；留空则直接跑CLIP图像分支"
    )
    parser.add_argument(
        "--freeze_clip",
        action="store_true",
        help="与训练时保持一致；通常如果训练时冻结过，这里也加上"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    bert = BertModel.from_pretrained("bert-base-uncased")
    model = catNet(bert, freeze_clip=args.freeze_clip).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    clip_features = None
    if args.precompute_clip:
        clip_features = load_clip_feature_file(args.precompute_clip)
        print(f"Loaded {len(clip_features)} precomputed CLIP features.")

    test(
        model=model,
        device=device,
        img_path=args.image,
        txt=args.text,
        score_label=args.label,
        clip_features=clip_features,
    )


if __name__ == "__main__":
    main()