import os

import torch
from torch.utils.data import Dataset

import numpy as np

import scipy.io as sio
from scipy.io import wavfile
from scipy import signal

N_BINS = 64
fs = 16000

# Dataset Extraction
def load_rir(filepath, target_fs):
    mat_data = sio.loadmat(filepath)
    rir = mat_data['h_air'].ravel()
    fs = mat_data['air_info']['fs'][0][0][0][0]
    if (fs > target_fs):
        fsx = fs // target_fs
        rir = rir[0::fsx]
    return rir

def get_direct_rir(rir, fs):
    num_sample = int((0.004 * fs) - 1)
    return rir[:num_sample]

def apply_reverberation(x, rir):
    rev_signal = signal.convolve(x, rir)
    return rev_signal[:len(x)]

def normalize(spect):
    feature_vector = spect.ravel()
    min_x = np.min(feature_vector)
    max_x = np.max(feature_vector)
    spect = 2*((spect - min_x)/(max_x-min_x)) - 1
    return spect, min_x, max_x

def add_noise_at_20db(spect):
    spect[spect == 0] = 1e-40
    return spect

def create_spectrogram(file, rir, norm=True):
    fs, x = wavfile.read(file)
    x_rev = apply_reverberation(x, rir)
    window_len = N_BINS*2
    hamming_window = signal.windows.hamming(window_len)
    f, t, stft_out = signal.stft(x_rev, fs, window=hamming_window, nfft=N_BINS*2, nperseg=N_BINS*2, noverlap=window_len / 2)
    num_extra_bands = stft_out.shape[0] - N_BINS
    stft_out = stft_out[:-num_extra_bands, :] if num_extra_bands > 0 else stft_out
    spect = np.abs(stft_out)
    if norm:
        spect = np.ma.log(spect).filled(np.min(np.ma.log(spect).flatten()))
        spect, min_x, max_x = normalize(spect)
        return spect, np.angle(stft_out), [min_x, max_x]
    return spect, np.angle(stft_out), None

def spect_train_set_for_rir(directory, rir, num_files=5):
    data = np.zeros((N_BINS,1))
    for i, filename in enumerate(os.listdir(directory)):
        if i == num_files:
            break
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            spect, phase, _ = create_spectrogram(f, rir)
            data = np.concatenate((data, spect), axis=1)
        else:
            num_files += 1
    data = data[:, 1:-(data.shape[1] % N_BINS - 1)]
    N = (data.shape[1] // N_BINS)
    data_arr = np.split(data, N, axis=1)
    return data_arr, None

def spect_test_datapoint(file, rir):
    spect, phase, _ = create_spectrogram(file, rir)
    pad_amount = N_BINS - (spect.shape[1] % N_BINS)
    if pad_amount != 0:
        zero_pad = np.zeros((N_BINS, pad_amount))
        spect = np.concatenate((spect, zero_pad), axis=1)
        phase = np.concatenate((phase, zero_pad), axis=1)
    N = (spect.shape[1] // N_BINS)
    spect_list = np.split(spect, N, axis=1)
    phase_list = np.split(phase, N, axis=1)
    return spect_list, phase_list

def spect_test_set_for_rir(directory, rir, num_files=5):
    spect_data_arr = []
    phase_data_arr = []
    for i, filename in enumerate(os.listdir(directory)):
        if i == num_files:
            break
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            spect_list, phase_list = spect_test_datapoint(f, rir)
            spect_data_arr.extend(spect_list)
            phase_data_arr.extend(phase_list)
        else:
            num_files += 1
    return spect_data_arr, phase_data_arr

def calculate_srr(rev_spect, dir_spect):
    eps = 2.2204e-16
    power_dir = np.square(np.abs(dir_spect))
    power_rev = np.square(np.abs(rev_spect - dir_spect))
    srr = power_dir / (power_rev + eps)
    srr[srr==0] = eps
    return srr

def calculate_esnr(f, rir, dir_rir):
    _, sig = wavfile.read(f)
    rev_sig = apply_reverberation(sig, rir)
    dir_sig = apply_reverberation(sig, dir_rir)
    resid_sig = rev_sig - dir_sig
    esnr = 10 * np.log(np.sum(np.square(dir_sig)) / np.sum(np.square(resid_sig)))
    return esnr

def ibm_from_srr(srr_dB, t):
    ibm = np.where(srr_dB > t, 1, 0)
    ibm = np.nan_to_num(ibm)
    return ibm

def ibm_train_set_for_rir(directory, rir, dir_rir, num_files=5):
    data = np.zeros((N_BINS,1))
    for i, filename in enumerate(os.listdir(directory)):
        if i == num_files:
            break
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            rev_spect, _, _ = create_spectrogram(f, rir, norm=False)
            dir_spect, _, _ = create_spectrogram(f, dir_rir, norm=False)
            srr = calculate_srr(rev_spect, dir_spect)
            esnr = calculate_esnr(f, rir, dir_rir)
            srr_dB = 10 * np.log(srr);
            T = -6
            ideal_binary_mask = ibm_from_srr(srr_dB, T + esnr)
            data = np.concatenate((data, ideal_binary_mask), axis=1)
        else:
            num_files += 1
    data = data[:, 1:-(data.shape[1] % N_BINS - 1)]
    N = (data.shape[1] // N_BINS)
    ibm_arr = np.split(data, N, axis=1)
    return ibm_arr

def ibm_test_datapoint(file, rir, dir_rir):
    rev_spect, _, _ = create_spectrogram(file, rir, norm=False)
    dir_spect, _, _ = create_spectrogram(file, dir_rir, norm=False)
    srr = calculate_srr(rev_spect, dir_spect)
    esnr = calculate_esnr(file, rir, dir_rir)
    srr_dB = 10 * np.log(srr);
    T = -6
    ideal_binary_mask = ibm_from_srr(srr_dB, T + esnr)
    pad_amount = N_BINS - (ideal_binary_mask.shape[1] % N_BINS)
    if pad_amount != 0:
        one_pad = np.ones((N_BINS, pad_amount))
        ideal_binary_mask = np.concatenate((ideal_binary_mask, one_pad), axis=1)
    N = (ideal_binary_mask.shape[1] // N_BINS)
    ibm_list = np.split(ideal_binary_mask, N, axis=1)
    return ibm_list

def ibm_test_set_for_rir(directory, rir, dir_rir, num_files=5):
    ibm_arr = []
    for i, filename in enumerate(os.listdir(directory)):
        if i == num_files:
            break
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            ibm_list = ibm_test_datapoint(f, rir, dir_rir)
            ibm_arr.extend(ibm_list)
        else:
            num_files += 1
    return ibm_arr

def create_spect_set(directory, rir_directory, spect_function, num_files=5, num_rirs=3, fs=16000, mask=False, mask_function=None):
    X_arr = []
    y_arr = []
    for i, rir_filename in enumerate(os.listdir(rir_directory)):
        if i == num_rirs:
            break
        r = os.path.join(rir_directory, rir_filename)
        if os.path.isfile(r):
            rir = load_rir(r, fs)
            dir_rir = get_direct_rir(rir, fs)
            full_rev_arr, _ = spect_function(directory, rir, num_files=num_files)
            X_arr.extend(full_rev_arr)
            if mask:
                ibm_arr = mask_function(directory, rir, dir_rir, num_files=num_files)
                y_arr.extend(ibm_arr)
            else:
                dir_rev_arr, _ = spect_function(directory, dir_rir, num_files=num_files)
                y_arr.extend(dir_rev_arr)
    return X_arr, y_arr


# Custom Dataset Definition
class CustomCIDataset(Dataset):
    def __init__(self, feature_array, label_array):
        self.feature_array = feature_array
        self.label_array = label_array
        self.min_x = 0;
        self.max_x = 0;
        self.min_y = 0;
        self.max_y = 0;

    def __len__(self):
        return len(self.label_array)

    def __getitem__(self, idx):
        feature = torch.unsqueeze(self.feature_array[idx, :, :], 0)
        label = torch.unsqueeze(self.label_array[idx, :, :], 0)
        return feature, label
    
    def normalize(self):
        feature_vector = torch.flatten(self.feature_array)
        label_vector = torch.flatten(self.label_array)
        self.min_x = torch.min(feature_vector)
        self.max_x = torch.max(feature_vector)
        self.min_y = torch.min(label_vector)
        self.max_y = torch.max(label_vector)
        self.feature_array = 2*((self.feature_array - self.min_x)/(self.max_x-self.min_x)) - 1
        self.label_array = 2*((self.label_array - self.min_y)/(self.max_y-self.min_y)) - 1
    
    def scale_values(self):
        return [self.min_x, self.max_x, self.min_y, self.max_y]

def extract_dataset(directory, rir_directory, num_files, num_rirs, training_set=True, mask=False):
    if training_set:
        X, y = create_spect_set(directory, rir_directory, spect_train_set_for_rir,
                                num_files=num_files, num_rirs=num_rirs,
                                mask=mask, mask_function=ibm_train_set_for_rir)
    else:
        X, y = create_spect_set(directory, rir_directory, spect_test_set_for_rir,
                                num_files=num_files, num_rirs=num_rirs,
                                mask=mask, mask_function=ibm_test_set_for_rir)

    X = np.array(X)
    y = np.array(y)

    X_tensor = torch.Tensor(X)
    y_tensor = torch.Tensor(y)

    return CustomCIDataset(X_tensor, y_tensor)
    

if __name__=="__main__":
    # Dataset Extraction
    directory = "../Data/Speech_Files/Training/"
    rir_directory = "../Data/RIR_Files/24Training/"

    directory_test = "../Data/Speech_Files/Testing/"
    rir_directory_test = "../Data/RIR_Files/Testing/"

    train_dataset = extract_dataset(directory, rir_directory, 100, 24)
    test_dataset = extract_dataset(directory_test, rir_directory_test, 140, 1)
