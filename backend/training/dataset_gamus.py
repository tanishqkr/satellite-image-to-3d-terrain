"""PyTorch Dataset wrapping HuggingFace earthflow/GAMUS HDF5 files.

Fixes applied:
- Bug 1 fixed: Synchronized spatial augmentations (flip/rotate applied identically to RGB + depth)
- Issue 6 fixed: Shuffled tile selection for geographic diversity
"""
import os
import random
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files


class GAMUSDataset(Dataset):
    def __init__(self, split: str = "train", max_samples: int = 200, img_size: int = 512, cache_dir: str = "data/gamus"):
        self.img_size = img_size
        self.cache_dir = Path(cache_dir) / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initializing GAMUS dataset (split={split}, target_samples={max_samples})...")
        
        # Discover files from earthflow/GAMUS repository
        all_files = list_repo_files("earthflow/GAMUS", repo_type="dataset")
        prefix_rgb = f"images/{split}/"
        prefix_agl = f"heights/{split}/"
        
        rgb_files = sorted([f for f in all_files if f.startswith(prefix_rgb) and f.endswith("_RGB.h5")])
        
        # Shuffle tiles for geographic diversity (fixed seed for reproducibility)
        rng = random.Random(42)
        rng.shuffle(rgb_files)
        
        # Each 1024x1024 tile yields 4 non-overlapping 512x512 patches
        num_tiles_needed = int(np.ceil(max_samples / 4))
        tiles_to_download = rgb_files[:num_tiles_needed]
        
        self.samples = []
        
        for rgb_rel in tiles_to_download:
            base_name = Path(rgb_rel).name.replace("_RGB.h5", "")
            agl_rel = f"{prefix_agl}{base_name}_AGL.h5"
            
            try:
                rgb_path = hf_hub_download(
                    "earthflow/GAMUS", rgb_rel,
                    repo_type="dataset", local_dir=self.cache_dir.parent.parent
                )
                agl_path = hf_hub_download(
                    "earthflow/GAMUS", agl_rel,
                    repo_type="dataset", local_dir=self.cache_dir.parent.parent
                )
                
                with h5py.File(rgb_path, "r") as f_rgb, h5py.File(agl_path, "r") as f_agl:
                    rgb_full = np.array(f_rgb["image"])
                    agl_full = np.array(f_agl["image"])
                
                # Extract 4 quadrants (top-left, top-right, bottom-left, bottom-right)
                H, W = rgb_full.shape[:2]
                quadrants = [
                    (0, 512, 0, 512),
                    (0, 512, 512, 1024),
                    (512, 1024, 0, 512),
                    (512, 1024, 512, 1024),
                ]
                
                for r1, r2, c1, c2 in quadrants:
                    if len(self.samples) >= max_samples:
                        break
                    rgb_patch = rgb_full[r1:r2, c1:c2]
                    agl_patch = agl_full[r1:r2, c1:c2]
                    # Filter out nodata / empty patches
                    if np.nanmax(agl_patch) > 0.5:
                        self.samples.append((rgb_patch, agl_patch))
                        
            except Exception as e:
                print(f"Skipping {base_name}: {e}")
                continue
                
            if len(self.samples) >= max_samples:
                break

        print(f"Loaded {len(self.samples)} valid 512x512 patches from GAMUS.")

        # Color-only transforms applied to RGB only (no spatial ops here)
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_patch, agl_patch = self.samples[idx]

        # Convert to PIL for transforms
        rgb_pil = Image.fromarray(rgb_patch)

        # Clean depth patch (replace NaN with 0, clamp positive)
        agl_clean = np.nan_to_num(agl_patch, nan=0.0, posinf=100.0, neginf=0.0)
        agl_clean = np.maximum(agl_clean, 0.0).astype(np.float32)
        agl_tensor = torch.from_numpy(agl_clean)

        # --- Bug 1 FIX: Synchronized spatial augmentations ---
        # Apply identical spatial transforms to BOTH rgb and depth
        if random.random() > 0.5:
            rgb_pil = TF.hflip(rgb_pil)
            agl_tensor = torch.flip(agl_tensor, dims=[-1])  # flip width

        if random.random() > 0.5:
            rgb_pil = TF.vflip(rgb_pil)
            agl_tensor = torch.flip(agl_tensor, dims=[-2])  # flip height

        # Random 90-degree rotation (0, 90, 180, 270)
        k = random.randint(0, 3)
        if k > 0:
            rgb_pil = TF.rotate(rgb_pil, angle=k * 90, expand=False)
            agl_tensor = torch.rot90(agl_tensor, k=k, dims=[-2, -1])

        # Color jitter on RGB only (depth is unaffected by color changes)
        rgb_pil = self.color_jitter(rgb_pil)

        # Convert RGB to tensor and normalize
        rgb_tensor = TF.to_tensor(rgb_pil)  # [C, H, W] float32 in [0, 1]
        rgb_tensor = self.normalize(rgb_tensor)

        return rgb_tensor, agl_tensor

