import pandas as pd
import numpy as np
from scipy.signal import butter, sosfilt

# files


def load(path, end_s=None, fs=1000, skip=6):
    data = pd.read_csv(path,
                       skiprows=skip,
                       header=None,
                       delimiter='\t')
    if not end_s:
        end_s = int(len(data) / fs)

    timestamp = (data[0] - data[0][0]).to_numpy()[:end_s * fs]
    sig = data.to_numpy()[:end_s * fs, 1:]

    return timestamp, sig


# peak sorting


def peak_expand(peaks, window_width, max):
    expanded_list = []
    for peak in peaks:
        temp_list = list(range(peak - window_width // 2, peak + window_width // 2))
        if all(0 < x < max for x in temp_list):
            expanded_list.append(temp_list)
    return np.array(expanded_list)


def window_stack(data, window_size):
    n_windows = data.shape[0] // window_size
    windowed_data = \
        np.stack([data[i: i + window_size] for i in range(0,
                                                          n_windows * window_size,
                                                          window_size)])
    return windowed_data.transpose(1, 2)


def filter_windows(all_windows, num_std=3):
    '''shape: [n_window, window_width]'''
    window_to_keep = np.ones(all_windows.shape[0])
    avg = np.mean(all_windows, axis=0)
    std = np.std(all_windows, axis=0)
    for window_idx, window in enumerate(all_windows):
        for idx, val in enumerate(window):
            if np.abs(val - avg[idx]) > num_std * std[idx]:
                window_to_keep[window_idx] = 0
    return window_to_keep
