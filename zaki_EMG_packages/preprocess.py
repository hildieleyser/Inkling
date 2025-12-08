import numpy as np

def fix_length(signal: np.ndarray, target_length: int = 10000) -> np.ndarray:
    """
    Make sure the EMG has exactly `target_length` time samples.

    - If it's shorter  -> pad with zeros at the end
    - If it's longer   -> cut the extra samples off the end

    Input shape:
        (16, N_time)

    Output shape:
        (16, target_length)
    """
    channels, length = signal.shape

    if length == target_length:
        # already the right size
        return signal

    if length < target_length:
        # need to pad on the time axis (axis=1)
        pad_size = target_length - length
        # ((0,0) no pad on channels, (0, pad_size) pad at the end of time)
        return np.pad(signal, ((0, 0), (0, pad_size)), mode="constant")

    # length > target_length -> just cut off the end
    return signal[:, :target_length]


def reshape_for_model(signal: np.ndarray) -> np.ndarray:
    """
    Prepare EMG for the model.

    Convert from (16, time) to (1, time, 16):

    - model expects: (batch_size, time_steps, channels)
    - here batch_size = 1 (a single EMG example)

    Returns:
        np.ndarray of shape (1, time, 16)
    """
    # (16, time) -> (time, 16)
    signal = signal.T

    # add batch dimension: (1, time, 16)
    signal = signal[np.newaxis, ...]

    return signal.astype("float32")
