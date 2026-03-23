import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import BertTokenizer, BertModel
from scipy.stats import pearsonr, spearmanr

from dataset_ava import AVACaptionsDataset
from models import catNet
from losses import emd_loss, binary_accuracy


def safe_corr(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)

    if len(pred) < 2:
        return 0.0, 0.0
    if np.std(pred) < 1e-12 or np.std(target) < 1e-12:
        return 0.0, 0.0

    plcc = pearsonr(pred, target)[0]
    srcc = spearmanr(pred, target)[0]

    if np.isnan(plcc):
        plcc = 0.0
    if np.isnan(srcc):
        srcc = 0.0

    return float(plcc), float(srcc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--no_amp", action="store_true", help="禁用混合精度")
    parser.add_argument(
        "--parn_pretrained",
        type=str,
        default="",
        help="PARN属性编码器预训练权重路径；留空则随机初始化",
    )
    parser.add_argument(
        "--freeze_parn",
        action="store_true",
        help="冻结PARN属性编码器权重",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased")

    model = catNet(
        bert,
        parn_pretrained_path=args.parn_pretrained if args.parn_pretrained else None,
        freeze_parn=args.freeze_parn,
    ).to(device)
    start_epoch = 1

    train_ds = AVACaptionsDataset(args.train_csv, args.images_dir, tokenizer)
    val_ds = AVACaptionsDataset(args.val_csv, args.images_dir, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    criterion = emd_loss(dist_r=1)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    use_amp = (not args.no_amp) and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
            print(f"Loaded checkpoint from {args.checkpoint}, resume at epoch {start_epoch}")
        else:
            model.load_state_dict(ckpt, strict=False)
            print(f"Loaded raw state_dict from {args.checkpoint}")

    if start_epoch > args.epochs:
        print(f"Checkpoint already at epoch {start_epoch - 1}, nothing to train.")
        return

    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    bins_tensor = torch.arange(1, 11, dtype=torch.float32, device=device)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0

        for step, (image, text_ids, text_mask, y) in enumerate(train_loader, 1):
            image = image.to(device, non_blocking=True)
            text_ids = text_ids.to(device, non_blocking=True)
            text_mask = text_mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                out = model(image, text_ids, text_mask)
                loss = criterion(out, y) / args.accum_steps

            scaler.scale(loss).backward()

            if step % args.accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * args.accum_steps
            if step % 50 == 0:
                print(f"Epoch {epoch} Step {step}: loss={running / 50:.4f}")
                running = 0.0

        model.eval()
        val_loss = 0.0
        val_acc_sum = 0.0
        val_acc_batches = 0

        pred_means_all = []
        gt_means_all = []

        with torch.no_grad():
            for image, text_ids, text_mask, y in val_loader:
                image = image.to(device, non_blocking=True)
                text_ids = text_ids.to(device, non_blocking=True)
                text_mask = text_mask.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with autocast(enabled=use_amp):
                    out = model(image, text_ids, text_mask)

                batch_loss = criterion(out, y).item()
                val_loss += batch_loss

                batch_acc = binary_accuracy(out, y, bins=10).item()
                val_acc_sum += batch_acc
                val_acc_batches += 1

                pred_mean = torch.sum(out * bins_tensor, dim=1)
                gt_mean = torch.sum(y * bins_tensor, dim=1)

                pred_means_all.extend(pred_mean.cpu().numpy().tolist())
                gt_means_all.extend(gt_mean.cpu().numpy().tolist())

        val_loss /= max(1, len(val_loader))
        val_acc = val_acc_sum / max(1, val_acc_batches)
        val_plcc, val_srcc = safe_corr(pred_means_all, gt_means_all)

        print(
            f"Epoch {epoch}: "
            f"val_emd={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_plcc={val_plcc:.4f} | "
            f"val_srcc={val_srcc:.4f}"
        )

        ckpt_path = os.path.join(checkpoint_dir, f"ammnet_parn_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_emd": val_loss,
                "val_acc": val_acc,
                "val_plcc": val_plcc,
                "val_srcc": val_srcc,
            },
            ckpt_path,
        )
        print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()
