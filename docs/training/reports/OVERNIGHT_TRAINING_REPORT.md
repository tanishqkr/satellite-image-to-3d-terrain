# DepthWizard Overnight Training Report (100% GAMUS)

**Date**: 2026-09-10 08:38:09  
**Target Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (8.59 GB VRAM)  
**Total Patches**: 15348 ($512 \times 512$)  
**Batch Size**: 8  
**Total Epochs**: 12  
**Total Runtime**: 3.61 hours  

## Loss Convergence Summary
- **Initial Loss**: `1.2481`
- **Best Final Loss**: `1.6439`
- **Relative Improvement**: `-31.7%`

## Epoch Progression Table
| Epoch | Average Loss | Duration (min) | Head Learning Rate | Checkpoint Saved |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.6439 | 20.79 min | 5.00e-04 | ⭐ BEST |
| 2 | nan | 19.03 min | 5.00e-04 | - |
| 3 | nan | 18.18 min | 4.88e-04 | - |
| 4 | nan | 18.05 min | 4.52e-04 | - |
| 5 | nan | 17.63 min | 3.97e-04 | - |
| 6 | nan | 17.57 min | 3.27e-04 | - |
| 7 | nan | 17.58 min | 2.50e-04 | - |
| 8 | nan | 17.64 min | 1.73e-04 | - |
| 9 | nan | 17.51 min | 1.03e-04 | - |
| 10 | nan | 17.51 min | 4.77e-05 | - |
| 11 | nan | 17.5 min | 1.22e-05 | - |
| 12 | nan | 17.57 min | 0.00e+00 | - |

## Checkpoint Artifacts
1. **Best Model Checkpoint**: `backend/weights/best_model.pth`
2. **Production Inference Weights**: `backend/weights/final_model.pth`
3. **Periodic Snapshots**: `backend/weights/checkpoints`
4. **Structured Telemetry**: `docs/training/logs/training_full_overnight.jsonl`
5. **Raw Execution Log**: `docs/training/logs/training_full_overnight.log`
