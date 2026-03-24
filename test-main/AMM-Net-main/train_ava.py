import os
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import BertTokenizer, BertModel
from scipy.stats import pearsonr, spearmanr

from dataset_ava import AVACaptionsDataset
from models import catNet
from losses import emd_loss, binary_accuracy, pairwise_rank_loss


def safe_corr(pred, target):
    pred   = np.asarray(pred)
    target = np.asarray(target)
    if len(pred) < 2:
        return 0.0, 0.0
    if np.std(pred) < 1e-12 or np.std(target) < 1e-12:
        return 0.0, 0.0
    plcc = pearsonr(pred, target)[0]
    srcc = spearmanr(pred, target)[0]
    return float(0.0 if np.isnan(plcc) else plcc), \
           float(0.0 if np.isnan(srcc) else srcc)


def build_optimizer(model, lr_base):
    """Split parameters into 3 LR groups."""
    bert_ids = {id(p) for p in model.txt_enc.bert.parameters()}
    swin_ids = {id(p) for p in model.img_enc.parameters()}

    # PARN backbone/bottleneck (may be frozen; only include requires_grad)
    parn_slow, parn_slow_ids = [], set()
    for name, p in model.attr_enc.named_parameters():
        if not name.startswith("attr_projs") and p.requires_grad:
            parn_slow.append(p)
            parn_slow_ids.add(id(p))

    slow_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) in bert_ids | swin_ids | parn_slow_ids
    ]
    fast_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in bert_ids | swin_ids | parn_slow_ids
    ]
    return optim.AdamW(
        [
            {"params": slow_params, "lr": lr_base * 0.1},
            {"params": fast_params, "lr": lr_base},
        ],
        weight_decay=1e-4,
    )


def build_scheduler(optimizer, total_steps, warmup_steps):
    """Linear warmup → cosine decay, applied per optimizer step."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir",   type=str, required=True)
    parser.add_argument("--train_csv",    type=str, required=True)
    parser.add_argument("--val_csv",      type=str, required=True)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--accum_steps",  type=int,   default=4)
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--checkpoint",   type=str,   default="")
    parser.add_argument("--no_amp",       action="store_true", help="禁用混合精度")
    parser.add_argument("--log_interval", type=int,   default=50)
    # loss weights
    parser.add_argument("--lambda_mean",  type=float, default=0.2,
                        help="mean-MSE loss weight")
    parser.add_argument("--lambda_rank",  type=float, default=0.1,
                        help="pairwise ranking loss weight")
    # PARN
    parser.add_argument("--parn_pretrained", type=str, default="",
                        help="AMM-Net.pt 或 PARN state_dict 路径；留空随机初始化")
    parser.add_argument("--freeze_parn",  action="store_true",
                        help="冻结 PARN backbone+bottleneck（attr_projs 仍训练）")
    parser.add_argument("--parn_cache",   type=str, default="",
                        help="precompute_parn.py 生成的缓存文件路径；"
                             "提供后跳过训练时的 ResNet-50 前向")
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dev   = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp   = (not args.no_amp) and torch.cuda.is_available()
    print("Using device:", device)

    use_cache = bool(args.parn_cache and os.path.exists(args.parn_cache))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert      = BertModel.from_pretrained("bert-base-uncased")

    model = catNet(
        bert,
        parn_pretrained_path=args.parn_pretrained if args.parn_pretrained else None,
        freeze_parn=args.freeze_parn,
        use_parn_cache=use_cache,
    ).to(device)

    parn_cache_path = args.parn_cache if use_cache else None

    train_ds = AVACaptionsDataset(
        args.train_csv, args.images_dir, tokenizer,
        is_train=True, parn_cache_path=parn_cache_path,
    )
    val_ds = AVACaptionsDataset(
        args.val_csv, args.images_dir, tokenizer,
        is_train=False, parn_cache_path=parn_cache_path,
    )

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    criterion = emd_loss(dist_r=1)
    optimizer = build_optimizer(model, args.lr)
    scaler    = GradScaler(amp_dev, enabled=use_amp)

    # Warmup 5% of total optimiser steps, then cosine decay
    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = max(1, int(total_steps * 0.05))
    scheduler       = build_scheduler(optimizer, total_steps, warmup_steps)
    opt_step        = 0   # counts actual optimizer steps (after accum)
    start_epoch     = 1
    best_srcc       = -1.0

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
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "opt_step" in ckpt:
                opt_step = ckpt["opt_step"]
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
            if "best_srcc" in ckpt:
                best_srcc = ckpt["best_srcc"]
            print(f"Resumed from {args.checkpoint} at epoch {start_epoch}")
        else:
            model.load_state_dict(ckpt, strict=False)
            print(f"Loaded raw state_dict from {args.checkpoint}")

    if start_epoch > args.epochs:
        print("Already trained. Nothing to do.")
        return

    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    bins_t = torch.arange(1, 11, dtype=torch.float32, device=device)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            image, text_ids, text_mask, y, parn_g, parn_scores = batch
            image      = image.to(device, non_blocking=True)
            text_ids   = text_ids.to(device, non_blocking=True)
            text_mask  = text_mask.to(device, non_blocking=True)
            y          = y.to(device, non_blocking=True)
            parn_g     = parn_g.to(device, non_blocking=True)
            parn_scores= parn_scores.to(device, non_blocking=True)

            with autocast(amp_dev, enabled=use_amp):
                out = model(image, text_ids, text_mask,
                            parn_g if use_cache else None,
                            parn_scores if use_cache else None)

                # ── Combined loss ─────────────────────────────────────────
                pred_mean = (out * bins_t).sum(dim=1)
                gt_mean   = (y  * bins_t).sum(dim=1)

                loss_emd  = criterion(out, y)
                loss_mean = F.mse_loss(pred_mean, gt_mean)
                loss_rank = pairwise_rank_loss(pred_mean, gt_mean)
                loss = (loss_emd
                        + args.lambda_mean * loss_mean
                        + args.lambda_rank * loss_rank) / args.accum_steps

            scaler.scale(loss).backward()

            if step % args.accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                scheduler.step()

            running_loss += loss.item() * args.accum_steps
            if step % args.log_interval == 0:
                lr_now = optimizer.param_groups[-1]["lr"]
                print(f"Epoch {epoch} Step {step}: "
                      f"loss={running_loss / args.log_interval:.4f}  "
                      f"lr={lr_now:.2e}")
                running_loss = 0.0

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss, val_acc_sum, val_acc_n = 0.0, 0.0, 0
        pred_means_all, gt_means_all = [], []

        with torch.no_grad():
            for batch in val_loader:
                image, text_ids, text_mask, y, parn_g, parn_scores = batch
                image      = image.to(device, non_blocking=True)
                text_ids   = text_ids.to(device, non_blocking=True)
                text_mask  = text_mask.to(device, non_blocking=True)
                y          = y.to(device, non_blocking=True)
                parn_g     = parn_g.to(device, non_blocking=True)
                parn_scores= parn_scores.to(device, non_blocking=True)

                with autocast(amp_dev, enabled=use_amp):
                    out = model(image, text_ids, text_mask,
                                parn_g if use_cache else None,
                                parn_scores if use_cache else None)

                val_loss  += criterion(out, y).item()
                val_acc_sum += binary_accuracy(out, y, bins=10).item()
                val_acc_n   += 1

                pred_means_all.extend((out * bins_t).sum(1).cpu().numpy().tolist())
                gt_means_all.extend((y   * bins_t).sum(1).cpu().numpy().tolist())

        val_loss /= max(1, len(val_loader))
        val_acc   = val_acc_sum / max(1, val_acc_n)
        val_plcc, val_srcc = safe_corr(pred_means_all, gt_means_all)

        print(f"Epoch {epoch}: "
              f"val_emd={val_loss:.4f} | val_acc={val_acc:.4f} | "
              f"val_plcc={val_plcc:.4f} | val_srcc={val_srcc:.4f}")

        # ── Checkpointing ─────────────────────────────────────────────────
        ckpt_data = {
            "epoch": epoch, "opt_step": opt_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_srcc": best_srcc,
            "val_emd": val_loss, "val_acc": val_acc,
            "val_plcc": val_plcc, "val_srcc": val_srcc,
        }
        ckpt_path = os.path.join(checkpoint_dir, f"ammnet_epoch{epoch}.pt")
        torch.save(ckpt_data, ckpt_path)
        print("Saved:", ckpt_path)

        # Save best model by val_srcc
        if val_srcc > best_srcc:
            best_srcc = val_srcc
            ckpt_data["best_srcc"] = best_srcc
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(ckpt_data, best_path)
            print(f"★ New best SRCC={best_srcc:.4f} → saved best_model.pt")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
