import argparse
import os
import sys
import time
import datetime
import glob
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

import scipy.io as sio
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy import signal

import matplotlib.pyplot as plt
from pystoi import stoi
from pesq import pesq

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

def create_spectrogram(file, rir):
    fs, x = wavfile.read(file)
    x_rev = apply_reverberation(x, rir)
    window_len = N_BINS*2
    hamming_window = signal.windows.hamming(window_len)
    f, t, stft_out = signal.stft(x_rev, fs, window=hamming_window, nfft=N_BINS*2, nperseg=N_BINS*2, noverlap=window_len / 2)
    num_extra_bands = stft_out.shape[0] - N_BINS
    stft_out = stft_out[:-num_extra_bands, :] if num_extra_bands > 0 else stft_out
    spect = np.abs(stft_out)
    spect = np.ma.log(spect).filled(np.min(np.ma.log(spect).flatten()))
    spect, min_x, max_x = normalize(spect)
    return spect, np.angle(stft_out), [min_x, max_x]

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
    temp_list = np.split(spect, N, axis=1)
    temp_phase_list = np.split(phase, N, axis=1)
    return temp_list, temp_phase_list

def spect_test_set_for_rir(directory, rir, num_files=5):
    data_arr = []
    phase_data_arr = []
    for i, filename in enumerate(os.listdir(directory)):
        if i == num_files:
            break
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            temp_list, temp_phase_list = spect_test_datapoint(f, rir)
            data_arr.extend(temp_list)
            phase_data_arr.extend(temp_phase_list)
        else:
            num_files += 1
    return data_arr, phase_data_arr

def create_spect_set(directory, rir_directory, spect_function, num_files=5, num_rirs=3, fs=16000):
    X_arr = []
    y_arr = []
    for i, rir_filename in enumerate(os.listdir(rir_directory)):
        if i == num_rirs:
            break
        r = os.path.join(rir_directory, rir_filename)
        if os.path.isfile(r):
            rir = load_rir(r, fs)
            dir_rir = get_direct_rir(rir, fs)
            full_rev_arr, phase_data_arr = spect_function(directory, rir, num_files=num_files)
            dir_rev_arr, phase_data_arr = spect_function(directory, dir_rir, num_files=num_files)
            X_arr.extend(full_rev_arr)
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


def extract_dataset(directory, rir_directory, num_files, num_rirs):
    X, y = create_spect_set(directory, rir_directory, spect_train_set_for_rir, num_files=num_files, num_rirs=num_rirs)
    
    X_tensor = torch.Tensor(X)
    y_tensor = torch.Tensor(y)

    dataset = CustomCIDataset(X_tensor, y_tensor)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader
    

if __name__=="__main__":
    # Dataset Extraction
    directory = "../Data/Speech_Files/Training/"
    rir_directory = "../Data/RIR_Files/24Training/"

    directory_test = "../Data/Speech_Files/Testing/"
    rir_directory_test = "../Data/RIR_Files/Testing/"

    train_dataloader = extract_dataset(directory, rir_directory, 100, 24)
    test_dataloader = extract_dataset(directory_test, rir_directory_test, 140, 1)
