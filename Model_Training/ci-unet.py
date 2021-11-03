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
    num_sample = int(0.004 * fs)
    return rir[:num_sample]


def apply_reverberation(x, rir):
    rev_signal = signal.convolve(x, rir)
    return rev_signal[:len(x)]


def normalize(spect):
    feature_vector = spect.ravel()
    min_x = np.min(feature_vector)
    max_x = np.max(feature_vector)
    spect = 2*((spect - min_x)/(max_x-min_x)) - 1
    return spect


def create_spectrogram(file, rir):
    fs, x = wavfile.read(file)
    x_rev = apply_reverberation(x, rir)
    window_len = int(fs * 0.032)
    hamming_window = signal.windows.hamming(window_len)
    f, t, spect = signal.spectrogram(x_rev, fs, window=hamming_window, nfft=512, noverlap=window_len / 2)
    num_extra_bands = spect.shape[0] - 256
    spect = spect[:-num_extra_bands, :] if num_extra_bands > 0 else spect
    spect = np.ma.log(spect).filled(0)
    spect = normalize(spect)
    return spect


def spect_train_set_for_rir(directory, rir, num_files=5):
    data = np.zeros((256, 1))
    for i, filename in enumerate(os.listdir(directory)):
        if i == num_files:
            break
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            spect = create_spectrogram(f, rir)
            data = np.concatenate((data, spect), axis=1)
        else:
            num_files += 1
    data = data[:, 1:-(data.shape[1] % 256 - 1)]
    N = (data.shape[1] // 256)
    data_arr = np.split(data, N, axis=1)
    return data_arr


def spect_test_set_for_rir(directory, rir, num_files=5):
    data_arr = []
    for i, filename in enumerate(os.listdir(directory)):
        if i == num_files:
            break
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            spect = create_spectrogram(f, rir)
            pad_amount = 256 - (spect.shape[1] % 256)
            if pad_amount != 0:
                zero_pad = np.zeros((256, pad_amount))
                spect = np.concatenate((spect, zero_pad), axis=1)
            N = (spect.shape[1] // 256)
            temp_list = np.split(spect, N, axis=1)
            data_arr.extend(temp_list)
        else:
            num_files += 1
    return data_arr


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
            full_rev_arr = spect_function(directory, rir, num_files=num_files)
            dir_rev_arr = spect_function(directory, dir_rir, num_files=num_files)
            X_arr.extend(full_rev_arr)
            y_arr.extend(dir_rev_arr)
        else:
            num_rirs += 1
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
        self.down1 = CL(1, 64, 5)
        self.down2 = CBL(64, 128, 5)
        self.down3 = CBL(128, 256, 5)
        self.down4 = CBL(256, 512, 5)
        self.down5 = CBL(512, 512, 5)
        self.down6 = CBL(512, 512, 5)
        self.down7 = CBL(512, 512, 5)
        self.down8 = CBR(512, 512, 5)

        self.up1 = DCDR(512, 512, 5, out_pad=1)   # 2x2
        self.up2 = DCDR(1024, 512, 5, out_pad=1)  # 4x4
        self.up3 = DCDR(1024, 512, 5, out_pad=1)  # 8x8
        self.up4 = DCR(1024, 512, 5, out_pad=1)   # 16x16
        self.up5 = DCR(1024, 256, 5, out_pad=1)   # 32x32
        self.up6 = DCR(512, 128, 5, out_pad=1)    # 64x64
        self.up7 = DCR(256, 64, 5, out_pad=1)     # 128x128
        self.up8 = DCT(128, 1, 5)                 # 256x256

    def forward(self, x):
        x1 = self.down1(x)
        # print(x1.shape)
        x2 = self.down2(x1)
        # print(x2.shape)
        x3 = self.down3(x2)
        # print(x3.shape)
        x4 = self.down4(x3)
        # print(x4.shape)
        x5 = self.down5(x4)
        # print(x5.shape)
        x6 = self.down6(x5)
        # print(x6.shape)
        x7 = self.down7(x6)
        # print(x7.shape)
        x8 = self.down8(x7)
        # print(x8.shape)

        x9 = self.up1(x8, x7)
        # print(x9.shape)
        x10 = self.up2(x9, x6)
        # print(x10.shape)
        x11 = self.up3(x10, x5)
        # print(x11.shape)
        x12 = self.up4(x11, x4)
        # print(x12.shape)
        x13 = self.up5(x12, x3)
        # print(x13.shape)
        x14 = self.up6(x13, x2)
        # print(x14.shape)
        x15 = self.up7(x14, x1)
        # print(x15.shape)
        x16 = self.up8(x15)
        # print(x16.shape)
        return x16


# Custom Dataset Definition
class CustomCIDataset(Dataset):
    def __init__(self, feature_array, label_array):
        self.feature_array = feature_array
        self.label_array = label_array

    def __len__(self):
        return len(self.label_array)

    def __getitem__(self, idx):
        feature = torch.unsqueeze(self.feature_array[idx, :, :], 0)
        label = torch.unsqueeze(self.label_array[idx, :, :], 0)
        return feature, label

    def normalize(self):
        feature_vector = torch.flatten(self.feature_array)
        label_vector = torch.flatten(self.label_array)
        min_x = torch.min(feature_vector)
        max_x = torch.max(feature_vector)
        min_y = torch.min(label_vector)
        max_y = torch.max(label_vector)
        self.feature_array = 2*((self.feature_array - min_x)/(max_x-min_x)) - 1
        self.label_array = 2*((self.label_array - min_y)/(max_y-min_y)) - 1


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


# Model Training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = CI_Unet()
net = net.to(device)
if device == 'cuda':
    print("Train on GPU...")
else:
    print("Train on CPU...")

# Initial learning rate
INITIAL_LR = 0.01
# Momentum for optimizer.
MOMENTUM = 0.9
# Regularization
REG = 1e-3
# Total number of training epochs
EPOCHS = 10
# Learning rate decay policy.
DECAY_EPOCHS = 2
DECAY = 0.8


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),
                             lr=INITIAL_LR,
                             weight_decay=REG)

global_step = 0
current_learning_rate = INITIAL_LR

for i in range(0, EPOCHS):
    print(datetime.datetime.now())
    net.train()
    print("Epoch %d:" % i)

    total_examples = 0
    correct_examples = 0

    train_loss = 0
    train_acc = 0

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs = inputs.to(device)  # Copy inputs to device
        targets = targets.to(device)  # Copy targets to device

        optimizer.zero_grad()  # Zero the gradient of the optimizer

        outputs = net.forward(inputs)  # Forward pass to generate outputs

        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()  # Backward loss and compute gradient

        optimizer.step()  # Apply gradient
        # print([batch_idx, loss])
        train_loss += loss
        global_step += 1

    avg_loss = train_loss / (batch_idx + 1)
    print("Training loss: %.4f" % (avg_loss))
    print(datetime.datetime.now())

    # Validate on the validation dataset
    print("Validation...")

    net.eval()

    val_loss = 0
    with torch.no_grad():  # Disable gradient during validation
        x = enumerate(test_dataloader)
        for batch_idx, (inputs, targets) in x:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss

    avg_loss = val_loss / len(test_dataloader)
    print("Validation loss: %.4f" % (avg_loss))

    # Handle the learning rate scheduler.
    if i % DECAY_EPOCHS == 0 and i != 0:
        current_learning_rate = current_learning_rate * DECAY
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_learning_rate

print("Optimization finished.")

torch.save(net.state_dict(), "../Saved_Models/ci-unet.pt")

# Sample Checking
feature, label = testing_data.__getitem__(1)
net.eval()
feature = torch.unsqueeze(feature, 0)
with torch.no_grad():
    feature = feature.to(device)
    output = net(feature)


def scale(data):
    data += -torch.min(data) + 1e-5
    data /= 2
    return data


plt.figure(figsize=(30, 10), dpi=200)
plt.subplot(131)
plt.imshow(scale(feature[0, 0, :, :].cpu()))
plt.colorbar()

plt.subplot(132)
plt.imshow(scale(output[0, 0, :, :].cpu()))
plt.colorbar()

plt.subplot(133)
plt.imshow(scale(label[0, :, :].cpu()))
plt.colorbar()

plt.savefig('res-1.png')
