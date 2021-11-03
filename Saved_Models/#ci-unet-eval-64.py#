# Import Packages
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

# Model Definition
class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CBL, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lr = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        out = self.cnn(x)
        out = self.bn(out)
        out = self.lr(out)
        return out


class CL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CL, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=2)
        self.lr = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        out = self.cnn(x)
        out = self.lr(out)
        return out


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CBR, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.cnn(x)
        out = self.relu(out)
        return out


class DCDR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, out_pad=0):
        super(DCDR, self).__init__()
        self.dcnn = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=2, output_padding=out_pad)
        self.bn = nn.BatchNorm2d(out_channels*2)
        self.drop = torch.nn.Dropout2d(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        out = self.dcnn(x1)
        out = torch.cat([out, x2], dim=1)
        out = self.bn(out)
        out = self.drop(out)
        out = self.relu(out)
        return out


class DCR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, out_pad=0):
        super(DCR, self).__init__()
        self.dcnn = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=2, output_padding=out_pad)
        self.bn = nn.BatchNorm2d(out_channels*2)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        out = self.dcnn(x1)
        out = torch.cat([out, x2], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


class DCT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, out_pad=0):
        super(DCT, self).__init__()
        self.dcnn = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, output_padding=out_pad)
        self.tanh = nn.Tanh()

    def forward(self, x1):
        out = self.dcnn(x1)
        out = self.tanh(out)
        return out


class CI_Unet(nn.Module):
    def __init__(self):
        super(CI_Unet, self).__init__()
        self.down1 = CL(1, 64, 5)                # 32x32
        self.down2 = CBL(64, 128, 5)             # 16x16
        self.down3 = CBL(128, 256, 5)            # 8x8
        self.down4 = CBL(256, 512, 5)            # 4x4
        self.down5 = CBL(512, 512, 5)            # 2x2
        self.down6 = CBR(512, 512, 5)            # 1x1
        
        self.up1 = DCDR(512, 512, 5, out_pad=1)   # 2x2
        self.up2 = DCDR(1024, 512, 5, out_pad=1)  # 4x4
        self.up3 = DCR(1024, 256, 5, out_pad=1)   # 8x8
        self.up4 = DCR(512, 128, 5, out_pad=1)    # 16x16
        self.up5 = DCR(256, 64, 5, out_pad=1)     # 32x32
        self.up6 = DCT(128, 1, 5)                 # 64x64

    def forward(self, x):
        x1 = self.down1(x)
        #print(x1.shape)
        x2 = self.down2(x1)
        #print(x2.shape)
        x3 = self.down3(x2)
        #print(x3.shape)
        x4 = self.down4(x3)
        #print(x4.shape)
        x5 = self.down5(x4)
        #print(x5.shape)
        x6 = self.down6(x5)
        #print(x6.shape)
        
        x7 = self.up1(x6, x5)
        #print(x9.shape)
        x8 = self.up2(x7, x4)
        #print(x10.shape)
        x9 = self.up3(x8, x3)
        #print(x11.shape)
        x10 = self.up4(x9, x2)
        #print(x12.shape)
        x11 = self.up5(x10, x1)
        #print(x13.shape)
        x12 = self.up6(x11)
        #print(x14.shape)
        return x12


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

# Dataset Extraction
directory = "../Data/Speech_Files/Training/"
rir_directory = "../Data/RIR_Files/24Training/"

directory_test = "../Data/Speech_Files/Testing/"
rir_directory_test = "../Data/RIR_Files/Testing/"

X_train, y_train = create_spect_set(directory, rir_directory, spect_train_set_for_rir, num_files=100, num_rirs=24)
X_test, y_test = create_spect_set(directory, rir_directory_test, spect_test_set_for_rir, num_files=140, num_rirs=1)

X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train)

X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)

# Dataset Initilization
training_data = CustomCIDataset(X_train_tensor, y_train_tensor)
testing_data = CustomCIDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=1, shuffle=True)


# Load Model from Checkpoint
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = CI_Unet()
net = net.to(device)

net.load_state_dict(torch.load("./ci-unet-64.pt"))
net.eval()

# Sample Checking
i = 1
feature, label = testing_data.__getitem__(i)
feature = torch.unsqueeze(feature, 0)
with torch.no_grad():
    feature = feature.to(device)
    output = net(feature)

plt.figure(figsize=(30, 10), dpi=200)
plt.subplot(131)
plt.imshow(np.flipud(feature[0, 0, :, :].cpu()), vmin=-1, vmax=1, cmap=plt.get_cmap("jet"))
plt.colorbar()
plt.title("Reverberant Spectrogram (64 Bins)")
plt.xlabel("Time Domain")
plt.ylabel("Frequency Domain")

plt.subplot(132)
plt.imshow(np.flipud(output[0, 0, :, :].cpu()), vmin=-1, vmax=1, cmap=plt.get_cmap("jet"))
plt.colorbar()
plt.title("Predicted Spectrogram (64 Bins)")
plt.xlabel("Time Domain")
plt.ylabel("Frequency Domain")

plt.subplot(133)
plt.imshow(np.flipud(label[0, :, :].cpu()), vmin=-1, vmax=1, cmap=plt.get_cmap("jet"))
plt.colorbar()
plt.title("Direct Path Spectrogram (64 Bins)")
plt.xlabel("Time Domain")
plt.ylabel("Frequency Domain")

plt.savefig("res-eval-64-%d.png" % i)

# Signal Reconstruction
def apply_net_to_spect(net, spect, scl_vals):
    spect_norm_tensor = torch.unsqueeze(torch.unsqueeze(torch.Tensor(spect), 0), 0)
    with torch.no_grad():
        net_output = net(spect_norm_tensor.to(device))
    net_output = np.squeeze(net_output).cpu().numpy()
    net_output = np.nan_to_num(net_output)
    net_output = ((net_output+1)/2)*(scl_vals[1] - scl_vals[0]) + scl_vals[0]
    return net_output

def recreate_from_spect(net_spect, phase, fs=16000):
    net_stft = np.exp(net_spect) * np.exp(1.0j * phase)
    zero_pad = np.zeros((1, np.shape(net_stft)[1]))
    net_stft = np.concatenate((net_stft, zero_pad), 0)
    net_stft = np.concatenate((net_stft[:-1], np.conj(net_stft[-1:0:-1, :])))
    window_len = N_BINS*2
    hamming_window = signal.windows.hamming(window_len)
    t, reconstructed_signal = signal.istft(net_stft, fs=fs, window=hamming_window, nfft=N_BINS*2, nperseg=N_BINS*2, noverlap=window_len / 2, input_onesided=False)
    return reconstructed_signal

def recreate_signal_datapoint(file, rir, net, sig_len):
    spect, phase, scl_vals = create_spectrogram(file, rir)
    pad_amount = N_BINS - (spect.shape[1] % N_BINS)
    if pad_amount != 0:
        zero_pad = np.zeros((N_BINS, pad_amount))
        spect = np.concatenate((spect, zero_pad), axis=1)
        phase = np.concatenate((phase, zero_pad), axis=1)
    N = (spect.shape[1] // N_BINS)
    spect_list = np.split(spect, N, axis=1)
    phase_list = np.split(phase, N, axis=1)
    reconstructed_signal = []
    for i in range(N):
        net_spect = apply_net_to_spect(net, spect_list[i], scl_vals)
        reconstructed_signal.extend(recreate_from_spect(net_spect, phase_list[i]))
    if (len(reconstructed_signal) < sig_len):
        reconstructed_signal = np.pad(reconstructed_signal , (0, sig_len - len(reconstructed_signal)), 'constant')
    return reconstructed_signal[:sig_len]

def recreate_from_spect_set(directory, rir_directory, net, num_files=5, num_rirs=3, fs=16000):
    reconstructed_signals = []
    direct_path_signals = []
    full_rev_signals = []
    for i, rir_filename in enumerate(os.listdir(rir_directory)):
        if i == num_rirs:
            break
        r = os.path.join(rir_directory, rir_filename)
        if os.path.isfile(r):
            rir = load_rir(r, fs)
            dir_rir = get_direct_rir(rir, fs)
            for i, filename in enumerate(os.listdir(directory)):
                if i == num_files:
                    break
                f = os.path.join(directory, filename)
                if os.path.isfile(f):
                    dir_rir = get_direct_rir(rir, 16000)
                    fs, x = wavfile.read(f)
                    dir_path_signal = apply_reverberation(x, dir_rir)
                    full_rev_signal = apply_reverberation(x, rir)
                    sig_len = len(dir_path_signal)
                    direct_path_signals.append(dir_path_signal)
                    full_rev_signals.append(full_rev_signal[:sig_len])
                    reconstructed_signals.append(recreate_signal_datapoint(f, rir, net, sig_len))
                else:
                    num_files += 1
        else:
            num_rirs += 1
    return reconstructed_signals, direct_path_signals, full_rev_signals

# Inteligibility Evaluation
directory_test = "C:\\Users\\rapiduser\\jupyter-env\\CI_Lab\\Feature and Label Extraction\\Data\\Speech Files\\Testing\\"
rir_directory_test = "C:\\Users\\rapiduser\\jupyter-env\\CI_Lab\\Feature and Label Extraction\\Data\\RIR Files\\Testing\\"

reconstructed_signals, direct_path_signals, full_rev_signals = recreate_from_spect_set(directory, rir_directory, net, num_files=140, num_rirs=1)

stoi_rec_sum = 0
stoi_rev_sum = 0

pesq_rec_sum = 0
pesq_rev_sum = 0

for i in range(len(reconstructed_signals)):
    dir_path = direct_path_signals[i]
    rec = np.real(reconstructed_signals[i])
    full_rev = full_rev_signals[i]
    
    dir_path = dir_path/np.max(dir_path)
    rec = rec/np.max(rec)
    full_rev = full_rev/np.max(full_rev)
    if i == 0:
        plt.figure(figsize=(20,20))
        
        plt.subplot(3, 1, 1)
        plt.plot(full_rev)
        plt.xlabel("Samples")
        plt.ylabel("Normalized Amplitude")
        plt.title("Reverberant Signal")

        plt.subplot(3, 1, 2)
        plt.plot(rec)
        plt.xlabel("Samples")
        plt.ylabel("Normalized Amplitude")
        plt.title("Recreated Signal")
        
        plt.subplot(3, 1, 3)
        plt.plot(dir_path)
        plt.xlabel("Samples")
        plt.ylabel("Normalized Amplitude")
        plt.title("Direct Path Signal")
 
        plt.savefig("res-eval-64-rec-comp-%d.png" % i)

    stoi_rec_sum += stoi(dir_path, rec, fs)
    stoi_rev_sum += stoi(dir_path, full_rev, fs)

    pesq_rec_sum += pesq(fs, dir_path, rec, 'wb')
    pesq_rev_sum += pesq(fs, dir_path, full_rev, 'wb')
    
stoi_rec_avg = stoi_rec_sum / len(reconstructed_signals)
stoi_rev_avg = stoi_rev_sum / len(reconstructed_signals)

pesq_rec_avg = pesq_rec_sum / len(reconstructed_signals)
pesq_rev_avg = pesq_rev_sum / len(reconstructed_signals)

labels = ["Reverberant Signal", "Recreated Signal"]
colors = ["red", "blue"]
stoi_values = [stoi_rev_avg, stoi_rec_avg]
pesq_values = [pesq_rev_avg, pesq_rec_avg]
print(stoi_values)
print(pesq_values)

plt.figure(figsize=(20,10))

plt.subplot(1, 2, 1)
plt.bar(labels, stoi_values, color=colors)
plt.xlabel("Signal")
plt.ylabel("STOI Value")
plt.title("Mean STOI results (N = 140)")
plt.ylim((0.2, 0.8))

plt.subplot(1, 2, 2)
plt.bar(labels, pesq_values, color=colors)
plt.xlabel("Signal")
plt.ylabel("PESQ Value")
plt.title("Mean PESQ results (N = 140)")
plt.ylim((0.5, 1.5))

plt.savefig("intel-eval-64.png")
