"""
DepthWizard Training Script — v2 (bug-fixed + stronger).
Fine-tunes Depth Anything V2 ViT-S on GAMUS dataset.
Optimized for NVIDIA GeForce RTX 4060.

Fixes vs v1:
- Bug 2 fixed: Per-sample normalization (was batch-wide, contaminating gradients)
- Issue 3 fixed: Uses TRAIN_SAMPLES from config (no more MVP_SAMPLES=40 override)
- Issue 9: Epochs bumped to 25, samples to 400, gradient clipping added
- Issue 10: num_workers=2 (data in-memory, Windows __main__ guard present)
- Added linear warmup for first 2 epochs then cosine decay
- Added differential LR: backbone 1e-5, DPT head 5e-4
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import contextlib
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoModelForDepthEstimation

from training.losses import CombinedLoss
from training.dataset_gamus import GAMUSDataset
from app.config import (
    MODEL_ID, WEIGHTS_DIR, LOGS_DIR,
    TRAIN_SAMPLES, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR
)

# Override for this stronger training run
TRAIN_SAMPLES = 400   # up from 40
TRAIN_EPOCHS  = 25    # up from 10
WARMUP_EPOCHS = 2     # linear warmup before cosine decay
GRAD_CLIP     = 1.0   # max gradient norm


def train():
    assert torch.cuda.is_available(), "CUDA required for training!"
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 1. Dataset — use TRAIN_SAMPLES (400), not the old MVP_SAMPLES=40
    dataset = GAMUSDataset(split="train", max_samples=TRAIN_SAMPLES)
    loader = DataLoader(
        dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )

    # 2. Model
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    model.to(device)
    model.train()

    # 3. Differential LR: lower LR for pretrained backbone, higher for DPT decode head
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    head_params     = [p for n, p in model.named_parameters() if "backbone" not in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-5},   # pretrained DINOv2 — small LR
        {"params": head_params,     "lr": 5e-4},   # DPT head — larger LR
    ], weight_decay=0.01)

    # 4. Warmup then cosine: linear warmup for WARMUP_EPOCHS, cosine for rest
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS          # linear ramp 0 -> 1
        progress = (epoch - WARMUP_EPOCHS) / max(TRAIN_EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    criterion = CombinedLoss(alpha=0.5).to(device)

    # 5. Logging
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "training_v2.jsonl"

    best_loss = float("inf")
    total_start = time.time()

    print(f"\nStarting training: {len(loader)} batches/epoch, {TRAIN_EPOCHS} epochs")
    print(f"Samples: {len(dataset)} | Batch size: {TRAIN_BATCH_SIZE} | Warmup: {WARMUP_EPOCHS} epochs\n")

    for epoch in range(1, TRAIN_EPOCHS + 1):
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, targets) in enumerate(loader):
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            autocast_ctx = (
                autocast('cuda', dtype=torch.float16)
                if device.type == 'cuda'
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                outputs = model(pixel_values=images)
                pred = outputs.predicted_depth

                # Resize prediction to match target
                if pred.shape[-2:] != targets.shape[-2:]:
                    pred = nn.functional.interpolate(
                        pred.unsqueeze(1), size=targets.shape[-2:],
                        mode="bilinear", align_corners=False
                    ).squeeze(1)

                # Bug 2 FIX: Per-sample normalization (not batch-wide scalar)
                pred_min = pred.amin(dim=(-2, -1), keepdim=True)
                pred_max = pred.amax(dim=(-2, -1), keepdim=True)
                pred_norm = (pred - pred_min) / (pred_max - pred_min + 1e-8)

                tgt_min = targets.amin(dim=(-2, -1), keepdim=True)
                tgt_max = targets.amax(dim=(-2, -1), keepdim=True)
                target_norm = (targets - tgt_min) / (tgt_max - tgt_min + 1e-8)

                loss = criterion(pred_norm, target_norm)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                # Gradient clipping (Issue 9)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            epoch_loss += loss.item()

            log_entry = {
                "epoch": epoch, "batch": batch_idx,
                "loss": round(loss.item(), 5),
                "lr_backbone": optimizer.param_groups[0]["lr"],
                "lr_head":     optimizer.param_groups[1]["lr"],
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

        scheduler.step()
        avg_loss = epoch_loss / max(len(loader), 1)
        elapsed = time.time() - t0
        lr_b = optimizer.param_groups[0]["lr"]
        lr_h = optimizer.param_groups[1]["lr"]

        print(f"Epoch [{epoch:02d}/{TRAIN_EPOCHS}] Loss: {avg_loss:.4f}  "
              f"LR backbone={lr_b:.2e} head={lr_h:.2e}  Time: {elapsed:.1f}s", flush=True)

        # Save best checkpoint (includes optimizer state for resumability)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": best_loss,
            }, WEIGHTS_DIR / "best_model.pth")
            print(f"  -> Best model updated (loss: {best_loss:.4f})", flush=True)

    # Save final weights-only file (for inference loading)
    torch.save(model.state_dict(), WEIGHTS_DIR / "final_model.pth")

    total_time = (time.time() - total_start) / 60
    print(f"\n{'='*50}")
    print(f"Training complete in {total_time:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Weights saved to: {WEIGHTS_DIR / 'best_model.pth'}")
    print(f"Logs saved to: {log_file}")


if __name__ == "__main__":
    train()
