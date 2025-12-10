# api_EMG/fast.py

from pathlib import Path
import tempfile

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from zaki_EMG_packages.predict import predict_from_hdf5

# 0 = rest (NO), 1 = power (YES)
LABELS = {
    0: "rest (no)",
    1: "power (yes)",
}

app = FastAPI(title="EMG YES/NO API")

# Allow all origins for now (tighten in prod if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "EMG 4-channel CNN API is running",
    }


@app.post("/predict_emg")
async def predict_emg(
    file: UploadFile = File(...),
    dataset: str = Form("0"),  # HDF5 dataset key, e.g. "0", "1", "149"
):
    """
    Upload an EMG HDF5 file and get a YES/NO prediction.

    Request:
      - file: .hdf5 containing EMG trials
      - dataset: which dataset key inside HDF5 to use (string)

    Response:
      - result: 0 (rest) or 1 (power)
      - label: human-readable description
    """
    # Save upload to a temp file on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    # Run your full pipeline
    pred = predict_from_hdf5(tmp_path, dataset_name=dataset)

    return {
        "result": int(pred),
        "label": LABELS.get(int(pred), "unknown"),
    }

# api_EMG/fast.py (additions)

from typing import List
from pydantic import BaseModel
import numpy as np

from zaki_EMG_packages.model import get_model, get_scaler
from zaki_EMG_packages.preprocess import prepare_live_4ch

# existing LABELS and app setup stay unchanged ...


class LiveEMG(BaseModel):
    emg: List[List[float]]  # shape (4, N_time)


@app.post("/predict_emg_live")
async def predict_emg_live(data: LiveEMG):
    """
    Predict from live 4-channel EMG array.

    Request JSON:
      {
        "emg": [[ch0 samples...],
                [ch1 samples...],
                [ch2 samples...],
                [ch3 samples...]]
      }

    Returns:
      - result: 0 (rest) or 1 (power)
      - label: human-readable
    """
    arr = np.array(data.emg, dtype="float32")  # (4, N)

    scaler = get_scaler()
    x = prepare_live_4ch(arr, scaler=scaler)  # (1, T, 4)

    model = get_model()
    proba = float(model.predict(x)[0, 0])
    pred = int(proba >= 0.5)

    return {
        "result": pred,
        "label": LABELS.get(pred, "unknown"),
        "proba_power": proba,
    }
