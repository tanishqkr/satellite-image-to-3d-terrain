"""
DepthWizard: Production Server Launcher.
Starts FastAPI backend on http://127.0.0.1:8000.
"""
import sys
import os
from pathlib import Path

# Ensure backend root is in python path
backend_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(backend_dir))
os.environ["PYTHONUNBUFFERED"] = "1"

import uvicorn

if __name__ == "__main__":
    print(f"Starting DepthWizard Backend on http://127.0.0.1:8000...", flush=True)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, log_level="info", access_log=True)
