import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import h5py
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
import tempfile
from pathlib import Path

# custom packages
from zaki_EMG_packages.extract import load_emg
from zaki_EMG_packages.preprocess import fix_length, reshape_for_model
from zaki_EMG_packages.model import get_model
from zaki_EMG_packages.predict import predict_from_hdf5


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {'inkling': 'it is up'}



<<<<<<< HEAD
# http://127.0.0.1:8000/predict?EMG_signal_array(shape=1
@app.get("/predict")
def predict(
        pickup_datetime: str,  # 2014-07-06 19:18:00
        pickup_longitude: float,    # -73.950655
        pickup_latitude: float,     # 40.783282
        dropoff_longitude: float,   # -73.984365
        dropoff_latitude: float,    # 40.769802
        passenger_count: int
    ):      # 1
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
=======
# # http://127.0.0.1:8000/predict?data={xxxxxxxxxxxxxxxxxx}
# @app.post("/predict")
# def predict(data: str, dataset: str = "0"):
#     """
#     Predict EMG gesture from an HDF5 file.

#     Params:
#         data: full path to the .hdf5 file
#         dataset: dataset name inside file (default "0")

#     Returns:
#         {"result": 0 or 1}
#     """

#     # Load raw EMG
#     signal = load_emg(data, dataset_name=dataset)

#     # Preprocess
#     signal = fix_length(signal, target_length=10000)
#     x = reshape_for_model(signal)

#     # Predict using loaded model
#     model = get_model()
#     proba = model.predict(x)[0][0]
#     label = int(proba >= 0.5)

#     return {"result": label}

GESTURE_LABELS = {
    0: "Open Gesture (NO)",
    1: "Power Gesture (YES)"
}

@app.get("/")
def root():
    return {"message": "EMG API is running 🎉"}


# ---- Path-based prediction (for testing) ----
@app.post("/predict")
def predict(data: str, dataset: str = "0"):
    path = Path(data)
    result = predict_from_hdf5(path, dataset_name=dataset)
    return {
        "result": result,
        "gesture": GESTURE_LABELS[result]
    }


# ---- File upload prediction (for real user) ----
@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...), dataset: str = "0"):

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    # Run prediction
    result = predict_from_hdf5(tmp_path, dataset_name=dataset)

    return {
        "filename": file.filename,
        "result": result,
        "gesture": GESTURE_LABELS[result]
    }
>>>>>>> d939f221b19436ae635c253ac73973de07f8f4fa
