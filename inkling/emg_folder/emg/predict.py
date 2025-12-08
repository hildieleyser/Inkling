"""
Prediction helper for the EMG Conv1D classifier.
"""

import numpy as np
import tensorflow as tf


def predict_emg(
    model: tf.keras.Model,
    X_batch,
    threshold: float = 0.5
):
    """
    Run binary predictions on a batch.

    Parameters
    ----------
    model : tf.keras.Model
        Trained EMG model.
    X_batch : np.ndarray
        Input batch of shape (batch, time, channels).
    threshold : float
        Decision threshold on sigmoid output.

    Returns
    -------
    y_prob : np.ndarray
        Probabilities in [0, 1].
    y_pred : np.ndarray
        Binary predictions (0/1).
    """
    y_prob = model.predict(X_batch)
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred
