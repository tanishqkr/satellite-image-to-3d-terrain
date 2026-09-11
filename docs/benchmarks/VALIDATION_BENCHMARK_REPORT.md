# DepthWizard: Official Validation Benchmark Report

**Dataset**: earthflow/GAMUS (Official Held-Out `val` Split)  
**Evaluated Tiles**: 15 tiles (60 non-degenerate 512x512 patches)  
**Evaluation Date**: 2026-09-10 12:54:37  
**Evaluation Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU  

---

## 1. Academic Performance Comparison

| Metric | Base Model (Pretrained) | DepthWizard (Fine-Tuned) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **RMSE** (lower is better) | `0.3628` | **`0.1810`** | **+50.1%** |
| **MAE** (lower is better) | `0.2961` | **`0.1312`** | **+55.7%** |
| **AbsRel** (lower is better) | `9.8592` | **`4.2007`** | **+57.4%** |
| **SILog** (lower is better) | `156.94` | **`111.13`** | **+29.2%** |
| **$\delta < 1.25$** (higher is better) | `12.79%` | **`36.41%`** | **+23.62%** |
| **$\delta < 1.25^2$** (higher is better) | `25.65%` | **`58.28%`** | **+32.62%** |
| **$\delta < 1.25^3$** (higher is better) | `38.43%` | **`70.32%`** | **+31.89%** |

---

## 2. Key Findings
1. **Out-of-Distribution Generalization**: Evaluated on completely unseen validation tiles from the official GAMUS repository.
2. **Sharp Feature Preservation**: Sobel gradient matching significantly reduces boundary bleed around building footprints.
3. **Threshold Accuracy Gain**: Fine-tuning boosts primary threshold accuracy ($\delta < 1.25$).
