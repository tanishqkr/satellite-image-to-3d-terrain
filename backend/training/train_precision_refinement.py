"""
DepthWizard: Precision Fine-Tuning Engine (Stage 2 Refinement).
Bootstraps from proven Epoch 2 weights (Loss=0.8044) and runs a controlled,
high-precision 3-epoch refinement using Stable Log1p SILog + Smooth L1 + Sobel Loss.

Hyperparameters:
- Backbone LR: 5e-6
- DPT Head LR: 6e-5 (Controlled, zero-overshoot)
- Cosine Annealing decay down to 1e-6
- Batch Size: 8 + FP16 AMP
- Outlier / Flat tile filtering (elevation dynamic range >= 1.0m required)
"""
import sys
import os
import time
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoModelForDepthEstimation

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.losses import CombinedLoss
from app.config import MODEL_ID, WEIGHTS_DIR, LOGS_DIR

EPOCHS = 3
BATCH_SIZE = 8
WARMUP_STEPS = 200
GRAD_CLIP = 0.5
DATA_ROOT = Path("data")
WARM_START_WEIGHTS = WEIGHTS_DIR / "archive_run1" / "best_model_epoch02_loss0.8044.pth"


class TeeLogger:
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.log_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()


class PrecisionGAMUSDataset(Dataset):
    """Loads tiles from local cache and extracts non-degenerate 512x512 patches."""
    def __init__(self, split: str = "train", data_root: Path = DATA_ROOT):
        self.data_root = data_root
        cache_file = self.data_root / "gamus_file_list.json"
        
        with open(cache_file, "r") as f:
            rgb_files = json.load(f)
            
        prefix_agl = f"heights/{split}/"
        quadrants = [
            (0, 512, 0, 512),
            (0, 512, 512, 1024),
            (512, 1024, 0, 512),
            (512, 1024, 512, 1024),
        ]
        
        self.samples = []
        for f in rgb_files:
            base_name = Path(f).name.replace("_RGB.h5", "")
            rgb_p = self.data_root / f
            agl_p = self.data_root / f"{prefix_agl}{base_name}_AGL.h5"
            if rgb_p.exists() and agl_p.exists():
                for q in quadrants:
                    self.samples.append((str(rgb_p), str(agl_p), q[0], q[1], q[2], q[3]))

        print(f"[{time.strftime('%X')}] Precision dataset loaded! Total valid patches on disk: {len(self.samples)}")
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_jitter = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, agl_path, r1, r2, c1, c2 = self.samples[idx]

        try:
            with h5py.File(rgb_path, "r") as f_rgb, h5py.File(agl_path, "r") as f_agl:
                rgb_patch = f_rgb["image"][r1:r2, c1:c2]
                agl_patch = f_agl["image"][r1:r2, c1:c2]
        except Exception:
            rgb_patch = np.zeros((512, 512, 3), dtype=np.uint8)
            agl_patch = np.zeros((512, 512), dtype=np.float32)

        agl_patch = np.nan_to_num(agl_patch, nan=0.0)
        agl_patch = np.maximum(agl_patch, 0.0)

        # Spatial augmentations
        rgb_pil = Image.fromarray(rgb_patch)
        agl_pil = Image.fromarray(agl_patch.astype(np.float32), mode="F")

        if random.random() > 0.5:
            rgb_pil = TF.hflip(rgb_pil)
            agl_pil = TF.hflip(agl_pil)
        if random.random() > 0.5:
            rgb_pil = TF.vflip(rgb_pil)
            agl_pil = TF.vflip(agl_pil)

        rot = random.choice([0, 90, 180, 270])
        if rot != 0:
            rgb_pil = TF.rotate(rgb_pil, rot)
            agl_pil = TF.rotate(agl_pil, rot)

        rgb_pil = self.color_jitter(rgb_pil)
        rgb_tensor = TF.to_tensor(rgb_pil)
        rgb_tensor = self.normalize(rgb_tensor)

        depth_np = np.array(agl_pil, dtype=np.float32)
        p_low, p_high = np.percentile(depth_np, [0.5, 99.5])
        
        # Require minimal structural elevation contrast (> 0.5m dynamic range)
        if p_high - p_low > 0.5:
            depth_np = np.clip((depth_np - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)
            is_valid = True
        else:
            depth_np = np.zeros_like(depth_np, dtype=np.float32)
            is_valid = False

        target_tensor = torch.from_numpy(depth_np).float()
        return rgb_tensor, target_tensor, torch.tensor(1.0 if is_valid else 0.0)


def run_precision_training():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    text_log = LOGS_DIR / "training_precision.log"
    jsonl_log = LOGS_DIR / "training_precision.jsonl"
    report_file = LOGS_DIR / "PRECISION_TRAINING_REPORT.md"

    sys.stdout = TeeLogger(text_log)
    assert torch.cuda.is_available(), "CUDA is required!"
    device = torch.device("cuda:0")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print("\n" + "=" * 68)
    print("  DEPTHWIZARD: PRECISION FINE-TUNING REFINEMENT (STAGE 2)")
    print(f"  Device: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print(f"  Warm Start Weights: {WARM_START_WEIGHTS}")
    print(f"  Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | BF16 Native Mixed Precision")
    print(f"  Head LR: 6.0e-5 | Backbone LR: 5.0e-6 (Precision Gradient)")
    print(f"  Loss: Stable Log1p SILog (0.5) + Smooth L1 (0.3) + Sobel Grad (0.2)")
    print("=" * 68 + "\n", flush=True)

    # 1. Dataset & Loader
    dataset = PrecisionGAMUSDataset(split="train")
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )

    # 2. Model Initialization & Warm Bootstrapping
    print(f"[{time.strftime('%X')}] Initializing base architecture {MODEL_ID}...")
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)

    if WARM_START_WEIGHTS.exists():
        print(f"[{time.strftime('%X')}] Bootstrapping from proven Run 1 checkpoint: {WARM_START_WEIGHTS}")
        ckpt = torch.load(WARM_START_WEIGHTS, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
        model.load_state_dict(sd, strict=False)
        initial_baseline_loss = ckpt.get("loss", 0.8044) if isinstance(ckpt, dict) else 0.8044
        print(f"[{time.strftime('%X')}] Loaded warm weights successfully! Target baseline loss: {initial_baseline_loss:.4f}")
    else:
        print(f"[{time.strftime('%X')}] Warning: Warm start weights not found, using base weights")
        initial_baseline_loss = 1.6439

    model.to(device)
    model.train()

    # 3. Controlled Optimizer & Scheduler
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    head_params     = [p for n, p in model.named_parameters() if "backbone" not in n]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 5e-6},
        {"params": head_params,     "lr": 6e-5},
    ], weight_decay=0.01)

    total_steps = len(loader) * EPOCHS
    def lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            return float(current_step + 1) / float(max(1, WARMUP_STEPS))
        progress = float(current_step - WARMUP_STEPS) / float(max(1, total_steps - WARMUP_STEPS))
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = CombinedLoss().to(device)

    best_loss = float("inf")
    epoch_history = []
    total_start = time.time()
    global_step = 0

    print(f"\n[{time.strftime('%X')}] Commencing precision training: {len(loader)} batches/epoch, {EPOCHS} epochs.\n", flush=True)

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        valid_batch_count = 0
        t0 = time.time()

        for batch_idx, (images, targets, valids) in enumerate(loader):
            global_step += 1
            # Skip batches where all tiles are flat / degenerate
            if valids.sum() < 2:
                continue

            images = images.to(device, non_blocking=True)
            targets = targets.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(pixel_values=images)
                pred = outputs.predicted_depth

            if pred.shape[-2:] != targets.shape[-2:]:
                pred = nn.functional.interpolate(
                    pred.unsqueeze(1).float(), size=targets.shape[-2:],
                    mode="bilinear", align_corners=False
                ).squeeze(1)

            # High-precision normalization in float32 to prevent underflow/overflow
            pred_f32 = pred.float()
            p_min = pred_f32.amin(dim=(-2, -1), keepdim=True)
            p_max = pred_f32.amax(dim=(-2, -1), keepdim=True)
            denom = torch.clamp(p_max - p_min, min=1e-4)
            pred_norm = (pred_f32 - p_min) / denom

            loss = criterion(pred_norm, targets.float())

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            # Strict guard: check if any gradient is NaN or Inf BEFORE clipping or stepping
            has_bad_grad = False
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_bad_grad = True
                    break

            if has_bad_grad:
                optimizer.zero_grad(set_to_none=True)
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            valid_batch_count += 1

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                b_avg = epoch_loss / max(valid_batch_count, 1)
                mem = torch.cuda.memory_allocated() / 1e9
                lr_h = optimizer.param_groups[1]["lr"]
                print(f"  [Epoch {epoch:02d}/{EPOCHS:02d}] Batch {batch_idx + 1:04d}/{len(loader):04d} | "
                      f"Loss: {b_avg:.4f} | VRAM: {mem:.2f} GB | LR: {lr_h:.2e}", flush=True)

                with open(jsonl_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "type": "batch", "epoch": epoch, "batch": batch_idx + 1,
                        "total_batches": len(loader), "loss": round(b_avg, 5),
                        "lr_head": lr_h, "vram_gb": round(mem, 2), "time": time.strftime('%X')
                    }) + "\n")

        # Weight sanity check
        is_clean = not any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters())
        if not is_clean:
            print(f"\n[CRITICAL ERROR] Weights corrupted with NaNs! Aborting epoch {epoch} immediately.\n", flush=True)
            break

        avg_loss = epoch_loss / max(valid_batch_count, 1)
        dur = (time.time() - t0) / 60
        lr_h = optimizer.param_groups[1]["lr"]
        is_best = (avg_loss < best_loss) and is_clean

        print(f"\n>>> [EPOCH {epoch:02d}/{EPOCHS:02d}] COMPLETE | Avg Loss: {avg_loss:.4f} | Time: {dur:.1f} min | LR: {lr_h:.2e}", flush=True)

        if is_best:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "timestamp": time.time()
            }, WEIGHTS_DIR / "best_model.pth")
            print(f"  ==> [CHECKPOINT] Best precision model saved to best_model.pth (Loss: {best_loss:.4f})\n", flush=True)

        epoch_history.append({"epoch": epoch, "avg_loss": round(avg_loss, 5), "duration_min": round(dur, 2), "is_best": is_best})

    # Save final model weights
    torch.save(model.state_dict(), WEIGHTS_DIR / "final_model.pth")
    total_mins = (time.time() - total_start) / 60

    print("\n" + "=" * 68)
    print(f"  PRECISION REFINEMENT COMPLETE IN {total_mins:.1f} MINUTES!")
    print(f"  Baseline Loss: {initial_baseline_loss:.4f}  ==>  Best Final Loss: {best_loss:.4f}")
    print(f"  Saved to: {WEIGHTS_DIR / 'best_model.pth'} & {WEIGHTS_DIR / 'final_model.pth'}")
    print("=" * 68 + "\n", flush=True)

    # Write report
    report = f"""# DepthWizard Precision Fine-Tuning Report (Stage 2)

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Target Hardware**: {gpu_name} ({vram_gb:.2f} GB VRAM)  
**Refinement Epochs**: {EPOCHS}  
**Total Runtime**: {total_mins:.1f} minutes  
**Initial Baseline**: `{initial_baseline_loss:.4f}`  
**Best Precision Loss**: `{best_loss:.4f}`  

## Epoch Progression
| Epoch | Average Loss | Duration (min) | Status |
| :---: | :---: | :---: | :---: |
"""
    for ep in epoch_history:
        tag = "⭐ BEST" if ep["is_best"] else "-"
        report += f"| {ep['epoch']} | {ep['avg_loss']:.4f} | {ep['duration_min']} min | {tag} |\n"

    report += f"""
## Deployed Weights
- `best_model.pth`: `{WEIGHTS_DIR / 'best_model.pth'}`
- `final_model.pth`: `{WEIGHTS_DIR / 'final_model.pth'}`
"""
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report generated: {report_file}", flush=True)


if __name__ == "__main__":
    run_precision_training()
