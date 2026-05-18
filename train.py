"""
train.py — Training pipeline for the autoregressive UI component detection model.

Features:
  - Mixed Precision Training (torch.cuda.amp) when CUDA is available.
  - AdamW optimiser with cosine LR schedule and linear warmup.
  - ViT encoder frozen for the first FREEZE_EPOCHS; then unfrozen and added to optimiser.
  - Per-step logging of loss_coord, loss_class, loss_stop, loss_total.
  - Saves best checkpoint (lowest validation total loss) to checkpoints/best.pt.

Usage:
    python train.py
"""

import os
import math
import torch
from torch.optim import AdamW
# AMP: access via torch namespace (no submodule import → no Pylance unresolved-import warning)
# torch.cuda.amp.GradScaler  — used as an attribute at runtime
# torch.autocast             — top-level, stable since PyTorch 1.12, not deprecated
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import VisionUIDetector
from loss import compute_loss
from utils import save_checkpoint, load_checkpoint


# ── LR schedule: cosine decay with linear warmup ─────────────────────────────

def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Returns a lambda-LR scheduler (warmup then cosine)."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


# ── One training epoch ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, scheduler, device, epoch):
    model.train()
    running = {"total": 0.0, "coord": 0.0, "class": 0.0, "stop": 0.0}
    n_steps = 0

    pbar = tqdm(loader, desc=f"  train e{epoch}", leave=False)
    for batch in pbar:
        images       = batch["images"].to(device, non_blocking=True)       # [B, 3, H, W]
        coords       = batch["coords"].to(device, non_blocking=True)       # [B, T, 4]
        classes      = batch["classes"].to(device, non_blocking=True)      # [B, T]
        padding_mask = batch["padding_mask"].to(device, non_blocking=True) # [B, T]
        lengths      = batch["lengths"].to(device, non_blocking=True)      # [B]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            pred_coords, pred_logits, pred_stop = model(images, coords, classes)
            # pred_coords : [B, T+1, 4]
            # pred_logits : [B, T+1, num_classes]
            # pred_stop   : [B, T+1]

            loss_total, loss_coord, loss_class, loss_stop = compute_loss(
                pred_coords, pred_logits, pred_stop,
                coords, classes, lengths, padding_mask,
            )

        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running["total"] += loss_total.item()
        running["coord"] += loss_coord.item()
        running["class"] += loss_class.item()
        running["stop"]  += loss_stop.item()
        n_steps += 1

        pbar.set_postfix(
            total=f"{loss_total.item():.3f}",
            coord=f"{loss_coord.item():.3f}",
            cls=f"{loss_class.item():.3f}",
            stop=f"{loss_stop.item():.3f}",
        )

    avg = {k: v / max(n_steps, 1) for k, v in running.items()}
    return avg


# ── One validation epoch ──────────────────────────────────────────────────────

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    running = {"total": 0.0, "coord": 0.0, "class": 0.0, "stop": 0.0}
    n_steps = 0

    for batch in tqdm(loader, desc="  val  ", leave=False):
        images       = batch["images"].to(device, non_blocking=True)
        coords       = batch["coords"].to(device, non_blocking=True)
        classes      = batch["classes"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        lengths      = batch["lengths"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            pred_coords, pred_logits, pred_stop = model(images, coords, classes)
            loss_total, loss_coord, loss_class, loss_stop = compute_loss(
                pred_coords, pred_logits, pred_stop,
                coords, classes, lengths, padding_mask,
            )

        running["total"] += loss_total.item()
        running["coord"] += loss_coord.item()
        running["class"] += loss_class.item()
        running["stop"]  += loss_stop.item()
        n_steps += 1

    return {k: v / max(n_steps, 1) for k, v in running.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nBuilding dataloaders …")
    train_loader = get_dataloader("train", num_workers=config.NUM_WORKERS)
    val_loader   = get_dataloader("val",   num_workers=config.NUM_WORKERS)
    print(f"  train steps/epoch: {len(train_loader)}  |  val steps/epoch: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nInitialising VisionUIDetector …")
    model = VisionUIDetector().to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {trainable:,} trainable / {total:,} total")

    # ── Freeze ViT encoder ────────────────────────────────────────────────────
    if config.FREEZE_EPOCHS > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
        frozen_params = sum(p.numel() for p in model.encoder.parameters())
        print(f"Frozen ViT encoder ({frozen_params:,} params) for {config.FREEZE_EPOCHS} epochs")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    def make_optimizer():
        return AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
        )

    total_steps  = len(train_loader) * config.EPOCHS
    warmup_steps = max(1, int(total_steps * config.WARMUP_RATIO))

    optimizer = make_optimizer()
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    encoder_unfrozen = (config.FREEZE_EPOCHS == 0)

    print(f"\nStarting training for {config.EPOCHS} epochs …\n")

    for epoch in range(1, config.EPOCHS + 1):

        # Unfreeze encoder after FREEZE_EPOCHS and rebuild optimiser
        if not encoder_unfrozen and epoch > config.FREEZE_EPOCHS:
            print(f"  → Unfreezing ViT encoder at epoch {epoch}")
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer.add_param_group({
                "params": list(model.encoder.parameters()),
                "lr": config.LR,
                "weight_decay": config.WEIGHT_DECAY,
            })
            # LambdaLR uses strict zip(param_groups, base_lrs, lr_lambdas).
            # Adding a param group without syncing these lists causes a crash.
            scheduler.base_lrs.append(config.LR)
            scheduler.lr_lambdas.append(scheduler.lr_lambdas[0])
            encoder_unfrozen = True

        print(f"Epoch {epoch}/{config.EPOCHS}  (lr={scheduler.get_last_lr()[0]:.2e})")

        t = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, device, epoch)
        v = eval_one_epoch(model, val_loader, device)

        print(
            f"  Train  total={t['total']:.4f}  coord={t['coord']:.4f}"
            f"  class={t['class']:.4f}  stop={t['stop']:.4f}"
        )
        print(
            f"  Val    total={v['total']:.4f}  coord={v['coord']:.4f}"
            f"  class={v['class']:.4f}  stop={v['stop']:.4f}"
        )

        # Always overwrite latest
        save_checkpoint(
            model, optimizer, epoch, v["total"],
            os.path.join(config.CHECKPOINT_DIR, "latest.pt"),
        )

        # Save best by validation total loss
        if v["total"] < best_val_loss:
            best_val_loss = v["total"]
            save_checkpoint(
                model, optimizer, epoch, v["total"],
                os.path.join(config.CHECKPOINT_DIR, "best.pt"),
            )
            print(f"  ✓ New best saved  (val_loss={v['total']:.4f})")

        print()

    print(f"Training complete.  Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved → {config.CHECKPOINT_DIR}/best.pt")


if __name__ == "__main__":
    train()
