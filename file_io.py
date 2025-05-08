import os
import json
import h5py
import numpy as np
from scipy.io import savemat

def save_h5(data, path, sample_rate):
    with h5py.File(path, 'w') as f:
        for ch, values in data.items():
            ds = f.create_dataset(ch, data=values)
            ds.attrs['sample_rate'] = sample_rate

def save_mat(data, path):
    savemat(path, {k: v for k, v in data.items()})
from scipy.io import savemat  # Добавить импорт

def save_mat(data, path):
    savemat(path, {k: v for k, v in data.items()})
def save_analysis_results(results, path):
    if path.endswith('.json'):
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
    elif path.endswith('.h5'):
        with h5py.File(path, 'w') as f:
            for channel, data in results.items():
                grp = f.create_group(channel)
                grp.create_dataset("freq", data=data['freq'])
                grp.create_dataset("psd", data=data['psd'])
