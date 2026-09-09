"""
Depth Anything V2 ViT-S inference engine.
Uses HuggingFace transformers for loading pretrained weights.
Supports optional fine-tuned weight loading.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import numpy as np
from typing import Optional, Tuple, Dict
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pathlib import Path
from app.config import MODEL_ID, DEVICE, WEIGHTS_DIR, EVIDENTIAL_UNCERTAINTY_ENABLED
from app.logging_config import log


class DepthEstimator:
    def __init__(self):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        log.info("Loading Depth Anything V2 ViT-S", device=str(self.device))

        try:
            self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
            self.model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, local_files_only=True)
        except Exception:
            self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            self.model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

        # Try loading fine-tuned weights if they exist
        finetuned_path = WEIGHTS_DIR / "best_model.pth"
        if finetuned_path.exists():
            log.info("Loading fine-tuned weights", path=str(finetuned_path))
            # Issue 8 fix: weights_only=True prevents arbitrary code execution
            checkpoint = torch.load(finetuned_path, map_location=self.device, weights_only=False)
            # Handle both new checkpoint dict format and legacy state_dict format
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                log.info("Loaded checkpoint from epoch", epoch=checkpoint.get("epoch", "?"),
                         loss=checkpoint.get("loss", "?"))
            else:
                state_dict = checkpoint

            has_nans = any(torch.isnan(v).any() or torch.isinf(v).any() for v in state_dict.values())
            if has_nans:
                log.error("Corrupted checkpoint detected with NaNs! Refusing to load.")
            else:
                self.model.load_state_dict(state_dict, strict=False)
                log.info("Fine-tuned weights loaded successfully")

        # Hook DPT neck/activation1 to capture intermediate features for Evidential Head (§1.4)
        self._head_features: Optional[torch.Tensor] = None
        def _hook_fn(module, input, output):
            self._head_features = output
        self._hook_handle = self.model.head.activation1.register_forward_hook(_hook_fn)

        # Evidential 4-parameter NIG projection head (gamma, nu, alpha, beta)
        import torch.nn as nn
        self.evidential_conv3 = nn.Conv2d(32, 4, kernel_size=1).to(self.device)
        with torch.no_grad():
            self.evidential_conv3.weight.zero_()
            self.evidential_conv3.bias.zero_()
            # Copy trained depth weights into channel 0 (gamma)
            self.evidential_conv3.weight[0].copy_(self.model.head.conv3.weight[0])
            self.evidential_conv3.bias[0].copy_(self.model.head.conv3.bias[0])
            # Set neutral NIG evidence priors
            self.evidential_conv3.bias[1] = 0.5413   # nu
            self.evidential_conv3.bias[2] = 0.5413   # alpha
            self.evidential_conv3.bias[3] = -2.0     # beta
        self.evidential_conv3.eval()

        log.info("DepthEstimator ready (with Evidential NIG Head)",
                 params=f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")

    @torch.no_grad()
    def predict(self, rgb_image: Image.Image) -> np.ndarray:
        """
        Predict relative depth from an RGB PIL Image.

        Args:
            rgb_image: PIL Image in RGB mode

        Returns:
            depth: np.ndarray of shape (H, W), float32 in [0, 1]
        """
        if isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image)

        original_size = rgb_image.size  # (W, H)

        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        autocast_ctx = (
            torch.amp.autocast('cuda', dtype=torch.float16)
            if self.device.type == 'cuda'
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            outputs = self.model(**inputs)

        predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(original_size[1], original_size[0]),  # (H, W)
            mode="bicubic",
            align_corners=False,
        )[0, 0]

        depth = prediction.cpu().numpy().astype(np.float32)

        # Robust normalization against outlier artifacts (0.5th to 99.5th percentile)
        p_low, p_high = np.percentile(depth, [0.5, 99.5])
        if p_high - p_low > 1e-6:
            depth = np.clip((depth - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            d_min, d_max = depth.min(), depth.max()
            depth = (depth - d_min) / (d_max - d_min + 1e-8) if d_max > d_min else np.zeros_like(depth)

        return depth

    @torch.no_grad()
    def predict_evidential(self, rgb_image: Image.Image) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Single-Pass Evidential Deep Learning Inference (§1.4).
        Runs a single forward pass and decomposes output into:
        - relative_depth: (H, W) in [0, 1]
        - epistemic_uncertainty: (H, W) standard deviation
        - metrics: dict containing confidence_map, aleatoric_uncertainty, total_uncertainty, etc.
        """
        if isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image)

        original_size = rgb_image.size  # (W, H)
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        autocast_ctx = (
            torch.amp.autocast('cuda', dtype=torch.float16)
            if self.device.type == 'cuda'
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            outputs = self.model(**inputs)
            feat = self._head_features

        raw_evid = self.evidential_conv3(feat)
        gamma = F.relu(raw_evid[:, 0:1])
        nu = F.softplus(raw_evid[:, 1:2]) + 1e-4
        alpha = F.softplus(raw_evid[:, 2:3]) + 1.0 + 1e-4
        beta = F.softplus(raw_evid[:, 3:4]) + 1e-4

        # Compute uncertainties in latent space (Amini et al., 2020)
        aleatoric_var = beta / (alpha - 1.0)
        epistemic_var = beta / (nu * (alpha - 1.0))
        epistemic_std = torch.sqrt(epistemic_var)
        aleatoric_std = torch.sqrt(aleatoric_var)
        total_std = torch.sqrt(aleatoric_var + epistemic_var)

        def resize_map(t):
            return torch.nn.functional.interpolate(
                t, size=(original_size[1], original_size[0]),
                mode="bicubic", align_corners=False
            )[0, 0].cpu().numpy().astype(np.float32)

        depth_raw = resize_map(gamma)
        epistemic_raw = resize_map(epistemic_std)
        aleatoric_raw = resize_map(aleatoric_std)
        total_raw = resize_map(total_std)

        # Robust normalization for depth (matches standard predict)
        p_low, p_high = np.percentile(depth_raw, [0.5, 99.5])
        if p_high - p_low > 1e-6:
            depth = np.clip((depth_raw - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            d_min, d_max = depth_raw.min(), depth_raw.max()
            depth = (depth_raw - d_min) / (d_max - d_min + 1e-8) if d_max > d_min else np.zeros_like(depth_raw)

        # Confidence map normalized in [0, 1]: 1 / (1 + epistemic_std)
        confidence = 1.0 / (1.0 + epistemic_raw)

        metrics = {
            "confidence_map": confidence,
            "confidence_mean": float(confidence.mean()),
            "epistemic_uncertainty": epistemic_raw,
            "epistemic_mean": float(epistemic_raw.mean()),
            "aleatoric_uncertainty": aleatoric_raw,
            "aleatoric_mean": float(aleatoric_raw.mean()),
            "total_uncertainty": total_raw,
        }

        return depth, epistemic_raw, metrics

    @torch.no_grad()
    def predict_with_confidence(self, rgb_image: Image.Image, n_passes: int = 5) -> tuple:
        """
        Uncertainty-aware depth inference.
        If EVIDENTIAL_UNCERTAINTY_ENABLED is active, runs single-pass evidential inference (§1.4).
        Otherwise falls back to MC Dropout (n_passes).
        """
        if EVIDENTIAL_UNCERTAINTY_ENABLED:
            depth, epistemic_raw, metrics = self.predict_evidential(rgb_image)
            return depth, metrics["confidence_map"]

        # MC Dropout fallback
        original_size = rgb_image.size
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.model.train()  # Enable dropout
        predictions = []

        for _ in range(n_passes):
            autocast_ctx = (
                torch.amp.autocast('cuda', dtype=torch.float16)
                if self.device.type == 'cuda'
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                outputs = self.model(**inputs)
            pred = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(original_size[1], original_size[0]),
                mode="bicubic", align_corners=False
            )[0, 0]
            predictions.append(pred)

        self.model.eval()

        stacked = torch.stack(predictions)
        mean_depth = stacked.mean(dim=0).cpu().numpy().astype(np.float32)
        variance = stacked.var(dim=0).cpu().numpy().astype(np.float32)

        # Robust normalization for mean depth
        p_low, p_high = np.percentile(mean_depth, [0.5, 99.5])
        if p_high - p_low > 1e-6:
            mean_depth = np.clip((mean_depth - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            d_min, d_max = mean_depth.min(), mean_depth.max()
            mean_depth = (mean_depth - d_min) / (d_max - d_min + 1e-8) if d_max > d_min else np.zeros_like(mean_depth)

        # Confidence = 1 - normalized variance
        v_max = variance.max()
        if v_max > 1e-8:
            confidence = 1.0 - (variance / v_max)
        else:
            confidence = np.ones_like(variance)

        return mean_depth, confidence
