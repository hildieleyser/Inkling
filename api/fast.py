import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import h5py
import numpy as np

from taxifare.ml_logic.preprocessor import preprocess_features
from taxifare.ml_logic.registry import load_model


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# http://127.0.0.1:8000/predict?data={xxxxxxxxxxxxxxxxxx}
@app.post("/predict")
def predict(
    data #hdf5   
    ):
    """

    """

    # assume intake is hdf5
    file_path = data
    
    # extract signal from hdf5 - check prev code
    with h5py.File(file_path, "r") as f:
        list(f.keys())
    
    #convert into np.array
    
    
    #load the scalar
    
    
    # scalar.transform
    
    
    # load model
    
    
    # result = model.predict
    
    
    # return result as a dict {'result': 1}