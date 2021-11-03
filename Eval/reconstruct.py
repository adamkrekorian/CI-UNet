import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from scipy.io import wavfile
from scipy import signal

import Data.dataset as dataset

N_BINS = 64
fs = 16000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def apply_net_to_spect(net, spect, scl_vals, mask=False):
    spect_norm_tensor = torch.unsqueeze(torch.unsqueeze(torch.Tensor(spect), 0), 0)
    with torch.no_grad():
        net_output = net(spect_norm_tensor.to(device))
    net_output = np.squeeze(net_output).cpu().numpy()
    net_output = np.nan_to_num(net_output)
    if mask:
        net_output = spect * net_output
    net_output = ((net_output+1)/2)*(scl_vals[1] - scl_vals[0]) + scl_vals[0]
    return net_output

def recreate_from_spect(net_spect, phase, fs=16000):
    net_stft = np.exp(net_spect) * np.exp(1.0j * phase)
    zero_pad = np.zeros((1, np.shape(net_stft)[1]))
    net_stft = np.concatenate((net_stft, zero_pad), 0)
    net_stft = np.concatenate((net_stft[:-1], np.conj(net_stft[-1:0:-1, :])))
    window_len = N_BINS*2
    hamming_window = signal.windows.hamming(window_len)
    _, reconstructed_signal = signal.istft(net_stft, fs=fs, window=hamming_window, nfft=N_BINS*2, nperseg=N_BINS*2, noverlap=window_len / 2, input_onesided=False)
    return reconstructed_signal

def recreate_signal_datapoint(file, rir, net, sig_len, mask=False):
    spect, phase, scl_vals = dataset.create_spectrogram(file, rir)
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
        net_spect = apply_net_to_spect(net, spect_list[i], scl_vals, mask=mask)
        reconstructed_signal.extend(recreate_from_spect(net_spect, phase_list[i]))
    if (len(reconstructed_signal) < sig_len):
        reconstructed_signal = np.pad(reconstructed_signal , (0, sig_len - len(reconstructed_signal)), 'constant')
    return reconstructed_signal[:sig_len]

def recreate_from_spect_set(directory, rir_directory, net, num_files=5, num_rirs=3, fs=16000, mask=False):
    reconstructed_signals = []
    direct_path_signals = []
    full_rev_signals = []
    for i, rir_filename in enumerate(os.listdir(rir_directory)):
        if i == num_rirs:
            break
        r = os.path.join(rir_directory, rir_filename)
        if os.path.isfile(r):
            rir = dataset.load_rir(r, fs)
            dir_rir = dataset.get_direct_rir(rir, fs)
            for i, filename in enumerate(os.listdir(directory)):
                if i == num_files:
                    break
                f = os.path.join(directory, filename)
                if os.path.isfile(f):
                    dir_rir = dataset.get_direct_rir(rir, 16000)
                    fs, x = wavfile.read(f)
                    dir_path_signal = dataset.apply_reverberation(x, dir_rir)
                    full_rev_signal = dataset.apply_reverberation(x, rir)
                    sig_len = len(dir_path_signal)
                    direct_path_signals.append(dir_path_signal)
                    full_rev_signals.append(full_rev_signal[:sig_len])
                    reconstructed_signals.append(recreate_signal_datapoint(f, rir, net, sig_len, mask=mask))
                else:
                    num_files += 1
        else:
            num_rirs += 1
    return reconstructed_signals, direct_path_signals, full_rev_signals