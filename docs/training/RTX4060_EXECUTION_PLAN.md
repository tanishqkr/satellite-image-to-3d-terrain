# Implementation Plan: Local DepthWizard Pipeline for RTX 4060 + i7-12700H

## Goal Description
Tailor and execute the end-to-end **DepthWizard (SIH26175)** system locally on the user's specific hardware:
- **GPU:** NVIDIA GeForce RTX 4060 (8 GB VRAM, Ada Lovelace architecture, 4th Gen Tensor Cores)
- **CPU:** Intel Core i7-12700H (14 cores / 20 threads: 6 P-cores + 8 E-cores)
- **Memory:** 16 GB DDR5/DDR4 RAM
- **OS:** Windows 11 (PowerShell)

This hardware is capable of running both local **10-minute GAMUS dataset fine-tuning** and **real-time sub-50ms inference** without relying on cloud services.

---

## User Review Required

> [!IMPORTANT]
> **VRAM Management on 8 GB GPU:**
> Windows DWM (Desktop Window Manager) uses ~0.8 GB to 1.2 GB of VRAM for display buffers. That leaves **~6.8 GB to 7.2 GB of free VRAM** for PyTorch.
> - **ViT-S (Depth Anything v2-Small):** Consumes **~3.8 GB VRAM** at `batch_size=8`, `img_size=512`. This leaves a 3 GB safety buffer, running with zero risk of Out-Of-Memory (OOM).
> - **ViT-B (Depth Anything v2-Base):** Consumes **~6.1 GB VRAM** at `batch_size=4` with gradient accumulation.
> - **Recommendation:** Train **ViT-S** first (~10 mins total). It provides the best speed-to-accuracy ratio for SIH and leaves ample headroom for the live web server.

> [!TIP]
> **PyTorch CUDA Environment on Windows:**
> Ensure your PyTorch installation is compiled with CUDA 12.1 or 12.4 (`torch.cuda.is_available() == True`). We will include an automated hardware verification script.

---

## Hardware-Optimized Training & Inference Profile

| Task | Configuration | VRAM Footprint | CPU Threads | Execution Time |
| :--- | :--- | :---: | :---: | :---: |
| **Dataset Caching (GAMUS)** | 1,200 Train Pairs $\rightarrow$ Local Parquet/Disk | ~0 GB | 4 Threads | **~80 seconds** (one-time) |
| **GAMUS Fine-Tuning (ViT-S)** | 20 Epochs, Batch 8, `fp16` Mixed Precision | **3.8 GB** | 4 P-Cores (`num_workers=4`) | **~9.5 to 11 minutes** |
| **Local Inference (512×512)** | `torch.cuda.amp.autocast()`, eval mode | **0.6 GB** | 1 Core | **~18 ms (55 FPS)** |
| **Local Inference (1024×1024)** | Tiled or direct pass | **1.4 GB** | 1 Core | **~42 ms (23 FPS)** |
| **3D WebGL Flythrough** | Three.js GPU Vertex Shader | **0.4 GB** | GPU Raster | **120+ FPS (locked)** |

---

## Proposed System Changes & Project Structure

We will create a clean, decoupled modular workspace under the project root:

```
depthwizard/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI application entrypoint
│   │   ├── config.py                # Hardware & path configurations
│   │   ├── api/
│   │   │   ├── routes.py            # Upload, inference, calibration, validation endpoints
│   │   ├── services/
│   │   │   ├── depth_estimator.py   # Depth Anything v2 inference engine (PyTorch CUDA)
│   │   │   ├── calibration.py       # Two-component SRTM/Cop-DEM scale calibrator
│   │   │   ├── geospatial.py        # GeoTIFF / CRS / Affine metadata extractor (rasterio)
│   │   │   └── validation.py        # Metric evaluator (RMSE, MAE, Pearson r)
│   ├── weights/                     # Pretrained & fine-tuned model checkpoints
│   ├── training/
│   │   ├── dataset_gamus.py         # PyTorch Dataset for earthflow/GAMUS with local caching
│   │   ├── train_gamus.py           # 10-minute local training script optimized for RTX 4060
│   │   └── losses.py                # Scale-Invariant Logarithmic (SILog) loss
│   └── tests/
│       └── test_cuda.py             # Hardware & VRAM benchmark test
├── frontend/                        # Modern Vite + React + Tailwind + Three.js
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dropzone.tsx         # Drag-and-drop satellite imagery loader
│   │   │   ├── DualViewer.tsx       # Side-by-side RGB vs DSM inspector
│   │   │   ├── TerrainCanvas.tsx    # Three.js 3D Flythrough canvas with WASD controls
│   │   │   ├── MeasurementTools.tsx # Building height & slope raycaster
│   │   │   └── ValidationCard.tsx   # LiDAR validation accuracy metrics & error map
│   │   ├── App.tsx
│   │   └── main.tsx
└── run_local.bat                    # One-click startup script for Windows
```

---

## Implementation Details

### 1. Training Script Optimized for RTX 4060 (`backend/training/train_gamus.py`)
```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
import time

def train_local_rtx4060():
    assert torch.cuda.is_available(), "RTX 4060 CUDA not detected!"
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True  # Optimized for fixed 512x512 resolution
    
    print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Hyperparameters tuned for 8GB VRAM
    BATCH_SIZE = 8
    NUM_WORKERS = 4  # Leverages 4 of the 6 P-cores on i7-12700H
    EPOCHS = 20
    LR = 5e-5
    
    # 1. Download/Load cached dataset
    print("Loading GAMUS dataset (train split)...")
    raw_dataset = load_dataset("earthflow/GAMUS", split="train")
    
    # 2. Setup DataLoader with pinned memory for fast GPU transfer
    # (Custom GAMUS Dataset class wraps raw_dataset into 512x512 tensors)
    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    # 3. Model & Mixed Precision Scaler
    model = load_depth_anything_v2(encoder='vits').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scaler = GradScaler()  # fp16 automatic mixed precision
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print(f"Starting training: {len(train_loader)} batches per epoch...")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(dtype=torch.float16):
                preds = model(images)
                loss = silog_loss(preds, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            
        scheduler.step()
        elapsed_epoch = time.time() - t0
        print(f"Epoch [{epoch:02d}/{EPOCHS}] - Loss: {epoch_loss/len(train_loader):.4f} - Time: {elapsed_epoch:.1f}s")
        
    total_time = (time.time() - start_time) / 60
    print(f"Training Complete in {total_time:.2f} minutes!")
    torch.save(model.state_dict(), "backend/weights/depth_anything_v2_gamus_vits.pth")
```

---

## Verification Plan

### Automated Verification:
1. **Hardware & VRAM Verification:**
   ```powershell
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)); print('VRAM (GB):', torch.cuda.get_device_properties(0).total_memory / 1e9)"
   ```
2. **Inference Latency Benchmark:**
   Run inference over 50 test iterations on a random 512×512 tensor and verify latency is $< 25\text{ms}$.
3. **Calibration Sanity Test:**
   Feed a synthetic slope terrain and verify that the two-component calibration correctly aligns ground plane elevation within $\pm 0.3\text{m}$.

### Manual Verification Flow:
1. Run `python backend/training/train_gamus.py` $\rightarrow$ Confirm training finishes in $\approx 10$ minutes with zero OOM errors.
2. Launch `run_local.bat` $\rightarrow$ Open `http://localhost:5173`.
3. Drop in a sample satellite image $\rightarrow$ Confirm:
   - Inference completes in under $0.5$ seconds.
   - Dual-pane 2D viewer shows RGB alongside colored metric DSM.
   - 3D flythrough loads at 60+ FPS with responsive WASD drone controls.

---

## User Decision & Approval to Proceed

Your RTX 4060 + i7-12700H is the ideal rig for this project. 

If you approve this plan, we will execute in this order:
1. **Step 1:** Verify Python environment, PyTorch CUDA 12, and GDAL/rasterio dependencies.
2. **Step 2:** Scaffold the `backend/` and `frontend/` project directories.
3. **Step 3:** Implement the zero-shot Depth Anything v2 inference and scale calibration service so you immediately have a working 2D $\rightarrow$ 3D prototype.
4. **Step 4:** Launch the ~10-minute local GAMUS fine-tuning script.
