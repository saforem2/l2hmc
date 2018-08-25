import os
import pickle
import numpy as np
from .jackknife import block_resampling, jackknife_err


def calc_avg_vals_errors(data, num_blocks=50):
    """ 
    Calculate average values and errors of using block jackknife
    resampling method.

    Args:
        num_blocks (int): Number of blocks to use for block jackknife
        resampling.
    """
    arr = np.array(data)
    avg_val = np.mean(arr)
    avg_val_rs = []
    arr_rs = block_resampling(arr, num_blocks)
    for block in arr_rs:
        avg_val_rs.append(np.mean(block))
    error = jackknife_err(y_i=avg_val_rs,
                          y_full=avg_val,
                          num_blocks=num_blocks)
    return avg_val, error


def load_data(data_dir):
    """
    Load all data from `.npy` and `.pkl` files contained in data_dir into
    numpy arrays and dictionaries respectively.

    Args:
        data_dir (directory, str):
            Directory containing data to load.

    Returns:
        arrays_dict (dict):
            Dictionary containing data loaded from `.npy` files.
            keys (str): String containing the filename without extension. 
            values (np.ndarray): Array containing data contained in file.
        pkl_dict (dict):
            Dictionary containing data load from `.pkl` files.
            keys (str): String containing the filename without extension.
            values (dict): Dictionary loaded in from file.
    """
    if not data_dir.endswith('/'):
        data_dir += '/'

    files = os.listdir(data_dir)
    if files == []:
        print(f"data_dir is empty. exiting!")
        raise ValueError

    np_files = []
    pkl_files = []
    for file in files:
        if file.endswith('.npy'):
            np_files.append(data_dir + file)
        if file.endswith('.pkl'):
            pkl_files.append(data_dir + file)

    #  np_load = lambda file: np.load(data_dir + file)
    def get_name(file): return file.split('/')[-1][:-4]

    arrays_dict = {}
    for file in np_files:
        key = file.split('/')[-1][:-4]
        #  key = get_name(file)
        arrays_dict[key] = np.load(file)

    pkl_dict = {}
    for file in pkl_files:
        key = get_name(file)
        with open(file, 'rb') as f:
            pkl_dict[key] = pickle.load(f)

    return arrays_dict, pkl_dict
