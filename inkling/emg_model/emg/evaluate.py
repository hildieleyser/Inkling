"""
Evaluation helper for the EMG Conv1D classifier.
"""

from typing import Dict, Any
import tensorflow as tf


def evaluate_emg_model(model: tf.keras.Model, X_test, y_test) -> Dict[str, Any]:
    """
    Evaluate the model on a test set.

    Returns
    -------
    metrics : dict
        Dict mapping metric names to values.
    """
    results = model.evaluate(X_test, y_test, verbose=0)
    metrics = dict(zip(model.metrics_names, results))
    return metrics
