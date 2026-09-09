from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKEND_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = BACKEND_ROOT / "weights"
LOGS_DIR = BACKEND_ROOT / "logs"
EXPORTS_DIR = BACKEND_ROOT / "exports"

# Model config
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEVICE = "cuda"
INFERENCE_SIZE = 512  # pixels

# Training config
TRAIN_SAMPLES = 200
TRAIN_EPOCHS = 10
TRAIN_BATCH_SIZE = 8
TRAIN_LR = 5e-5
TRAIN_IMG_SIZE = 512

# Validation
MAX_UPLOAD_SIZE_MB = 100
MIN_IMAGE_DIM = 64
MAX_IMAGE_DIM = 8192
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Global DTM Fusion (§2.1)
import os
DTM_FUSION_ENABLED = os.getenv("DTM_FUSION_ENABLED", "true").lower() in ("true", "1", "yes")
DTM_SOURCE = os.getenv("DTM_SOURCE", "remote")  # "remote", "local_cache", "disabled"
DTM_CACHE_DIR = Path(os.getenv("DTM_CACHE_DIR", str(BACKEND_ROOT / "data" / "dtm_cache")))
DTM_CUTOFF_WAVELENGTH_M = float(os.getenv("DTM_CUTOFF_WAVELENGTH_M", "150.0"))

# Oblique & Off-Nadir Perspective Rectification (§2.2)
ORTHORECTIFY_ENABLED = os.getenv("ORTHORECTIFY_ENABLED", "true").lower() in ("true", "1", "yes")
OBLIQUE_THRESHOLD_DEG = float(os.getenv("OBLIQUE_THRESHOLD_DEG", "15.0"))

# Shadow-to-Scale Photometric Inversion (§1.1)
SHADOW_SCALE_ENABLED = os.getenv("SHADOW_SCALE_ENABLED", "true").lower() in ("true", "1", "yes")
SHADOW_MIN_STRUCTURES = int(os.getenv("SHADOW_MIN_STRUCTURES", "2"))

# Probabilistic Occupancy & Viewshed Ray Marching (§3.2, §4.1)
OCCUPANCY_UNKNOWN_ENABLED = os.getenv("OCCUPANCY_UNKNOWN_ENABLED", "true").lower() in ("true", "1", "yes")
BAYESIAN_P_OCC = float(os.getenv("BAYESIAN_P_OCC", "0.70"))
BAYESIAN_P_FREE = float(os.getenv("BAYESIAN_P_FREE", "0.40"))
BAYESIAN_L_MIN = float(os.getenv("BAYESIAN_L_MIN", "-5.0"))
BAYESIAN_L_MAX = float(os.getenv("BAYESIAN_L_MAX", "5.0"))

# Single-Pass Evidential Uncertainty & Risk Maps (§1.4)
EVIDENTIAL_UNCERTAINTY_ENABLED = os.getenv("EVIDENTIAL_UNCERTAINTY_ENABLED", "true").lower() in ("true", "1", "yes")
EVIDENTIAL_LAMBDA_REG = float(os.getenv("EVIDENTIAL_LAMBDA_REG", "0.05"))

# Automated 3D Battle Damage Assessment & Volumetric Differencing (§4.2)
BDA_ENABLED = os.getenv("BDA_ENABLED", "true").lower() in ("true", "1", "yes")
BDA_NOISE_THRESHOLD_M = float(os.getenv("BDA_NOISE_THRESHOLD_M", "0.20"))
BDA_BETA_MODE = os.getenv("BDA_BETA_MODE", "true").lower() in ("true", "1", "yes")
BDA_MIN_BUILDING_HEIGHT_M = float(os.getenv("BDA_MIN_BUILDING_HEIGHT_M", "2.0"))

# Autonomous NOE & Terrain-Following Path Planning (§4.3)
NOE_ROUTING_ENABLED = os.getenv("NOE_ROUTING_ENABLED", "true").lower() in ("true", "1", "yes")
NOE_MIN_CLEARANCE_M = float(os.getenv("NOE_MIN_CLEARANCE_M", "3.0"))
NOE_MAX_ALTITUDE_M = float(os.getenv("NOE_MAX_ALTITUDE_M", "35.0"))
NOE_WEIGHT_EXPOSURE = float(os.getenv("NOE_WEIGHT_EXPOSURE", "4.0"))
NOE_WEIGHT_UNKNOWN = float(os.getenv("NOE_WEIGHT_UNKNOWN", "6.0"))
NOE_WEIGHT_ALTITUDE = float(os.getenv("NOE_WEIGHT_ALTITUDE", "1.5"))
NOE_WEIGHT_DISTANCE = float(os.getenv("NOE_WEIGHT_DISTANCE", "1.0"))

