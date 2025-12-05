"""
Training utilities for the EMG Conv1D classifier.
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from .model import build_emg_model, compile_emg_model


def train_emg_model(
    X_train,
    y_train,
    input_shape,
    batch_size: int = 32,
    epochs: int = 100
):
    """
    Build, compile, and train the EMG model.

    Returns
    -------
    model : tf.keras.Model
        The trained model (with best weights from early stopping).
    history : tf.keras.callbacks.History
        Training history object.
    """
    model = build_emg_model(input_shape)
    model = compile_emg_model(model)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    return model, history
