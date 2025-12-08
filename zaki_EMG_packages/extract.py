from pathlib import Path
from typing import Union, List

import h5py          # to read .hdf5 files
import numpy as np   # to work with arrays

def list_datasets(h5_path: Union[str, Path]) -> List[str]:
    """
    List the top-level dataset names in an HDF5 file.

    Use this first if you don't know which key contains the EMG signal.
    """
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as f:
        # top-level keys (groups / datasets)
        return list(f.keys())


def load_emg(
    h5_path: Union[str, Path],
    dataset_name: str,
    transpose: bool = False,
) -> np.ndarray:
    """
    Load EMG signal from an HDF5 file and return it as a NumPy array.

    Parameters
    ----------
    h5_path : str or Path
        Path to the .hdf5 file.
    dataset_name : str
        Name of the dataset inside the file that contains the EMG signal.
        (use list_datasets(...) to discover it)
    transpose : bool, default False
        If True, returns the transpose of the array (useful if your
        data is (N, channels) and you prefer (channels, N), or vice versa).

    Returns
    -------
    np.ndarray
        EMG signal as a float32 NumPy array.
    """
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as f:
        data = f[dataset_name][()]  # read dataset into memory

    emg = np.asarray(data, dtype="float32")

    if transpose:
        emg = emg.T

    return emg
