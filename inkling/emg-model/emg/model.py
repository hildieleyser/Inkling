"""
Model definition and compilation for the Conv1D EMG classifier.
"""

import tensorflow as tf
from tensorflow.keras import layers, Sequential


def build_emg_model(input_shape):
    """
    Build the Conv1D architecture used in your EMG notebook.
    Example input_shape = (10000, 16)
    """
    model = Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv1D(32, kernel_size=7, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=4),

        # Block 2
        layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=4),

        # Block 3
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=4),

        layers.Flatten(),

        layers.Dense(64, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),

        # Binary output
        layers.Dense(1, activation='sigmoid')
    ])

    return model


def compile_emg_model(model):
    """
    Compile the model with binary loss and metrics.
    """
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model
