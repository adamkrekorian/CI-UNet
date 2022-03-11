import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy import signal

import matplotlib.pyplot as plt

import Data.dataset as ds

N_BINS = 64
fs = 16000
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def rescale_spect(spect, scl_vals):
    min_, max_ = scl_vals
    spect = ((spect + 1) / 2) * (max_ - min_) + min_ 
    spect = np.power(10, spect)
    return spect

def apply_net(net, spect):
    spect_norm_tensor = torch.unsqueeze(torch.unsqueeze(torch.Tensor(spect), 0), 0).to(device)
    with torch.no_grad():
        net_output = net(spect_norm_tensor)
    net_output = np.squeeze(net_output).cpu().numpy()
    net_output = np.nan_to_num(net_output)
    return net_output


def apply_net_to_spect(net, spect, scl_vals, mask=False, rescale=False):
    net_output = apply_net(net, spect)
    if mask:
        spect = rescale_spect(spect, scl_vals) if rescale else spect
        net_output = spect * net_output
    else:
        net_output = rescale_spect(net_output, scl_vals) if rescale else net_output
    return net_output


def apply_net_to_full_spect(net, spect, scl_vals, mask=False, rescale=False):
    pad_amount = N_BINS - (spect.shape[1] % N_BINS)
    spect = ds.zero_pad(spect, pad_amount)
    N = spect.shape[1] // N_BINS
    spect_list = np.split(spect, N, axis=1)
    full_signal_spect = np.zeros((np.shape(spect)[0], 1))
    for sp in spect_list:
        net_output = apply_net_to_spect(net, sp, scl_vals, mask, rescale)
        full_signal_spect = np.append(full_signal_spect, net_output, axis=1)
    full_signal_spect = full_signal_spect[:, 1:-pad_amount]
    return full_signal_spect


def pad_spect(spect):
    zero_pad = np.zeros((1, np.shape(spect)[1]))
    spect = np.concatenate((spect, zero_pad), 0)
    return spect

def recreate_from_spect(net_spect, phase, fs=16000):
    exp_phase = np.exp(1.0j * phase)
    net_stft = net_spect * exp_phase
    net_stft = pad_spect(net_stft)
    net_stft = np.concatenate((net_stft[:-1], np.conj(net_stft[-1:0:-1, :])))
    window_len = N_BINS*2
    hamming_window = signal.windows.hamming(window_len)
    _, rec_signal = signal.istft(net_stft, fs=fs, window=hamming_window,
                                 nfft=N_BINS*2, nperseg=N_BINS*2,
                                 noverlap=window_len / 2, input_onesided=False)
    return rec_signal

def load_bin_weights():
    bin_weight_path = "./Eval/bin_weights.csv"
    bin_weights = pd.read_csv(bin_weight_path, header=None).to_numpy()
    return bin_weights

def apply_bin_weights(net_spect, bin_weights):
    power_spect = np.multiply(net_spect, np.conj(net_spect))
    weighted_sum_spect = np.matmul(bin_weights, power_spect)
    spect_22 = np.abs(np.sqrt(weighted_sum_spect))
    return spect_22

def vocode_spect(spect_22):
    vocoded_signal = 0
    return vocoded_signal

def vocode_from_spect(net_spect, phase, fs=16000):
    net_stft = np.exp(net_spect) * np.exp(1.0j * phase)
    net_stft = pad_spect(net_stft)
    spect_22 = apply_bin_weights(net_stft)
    vocoded_signal = vocode_spect(spect_22)
    return vocoded_signal

def recreate_signal(file, rir, net, sig_len, mask=False):
    spect, phase, scl_vals = ds.create_spectrogram(file, rir)
    pad_amount = N_BINS - (spect.shape[1] % N_BINS)
    spect = ds.zero_pad(spect, pad_amount)
    phase = ds.zero_pad(phase, pad_amount)
    N = spect.shape[1] // N_BINS
    spect_list = np.split(spect, N, axis=1)
    phase_list = np.split(phase, N, axis=1)
    rec_signal = []
    for sp, ph in zip(spect_list, phase_list):
        if net is not None:
            sp = apply_net_to_spect(net, sp, scl_vals, mask=mask, rescale=True)
        else:
            sp = rescale_spect(sp, scl_vals)
        rec_signal.extend(recreate_from_spect(sp, ph))
    if (len(rec_signal) < sig_len):
        rec_signal = np.pad(rec_signal, (0, sig_len - len(rec_signal)), 'constant')
    return rec_signal[:sig_len]

def recreate_from_spect_set(directory, rir_directory, net, num_files=140, num_rirs=1, fs=16000, mask=False):
    reconstructed_signals = []
    direct_path_signals = []
    full_rev_signals = []
    for i, rir_filename in enumerate(os.listdir(rir_directory)):
        if i == num_rirs:
            break
        r = os.path.join(rir_directory, rir_filename)
        if os.path.isfile(r):
            rir = ds.load_rir(r, fs)
            dir_rir = ds.get_direct_rir(rir, fs)
            for i, filename in enumerate(os.listdir(directory)):
                if i == num_files:
                    break
                f = os.path.join(directory, filename)
                if os.path.isfile(f):
                    fs, x = wavfile.read(f)
                    sig_len = len(ds.apply_reverberation(x, dir_rir))
                    dir_path_signal = recreate_signal(f, dir_rir, None, sig_len, mask=mask)
                    full_rev_signal = recreate_signal(f, rir, None, sig_len, mask=mask)
                    rec_signal = recreate_signal(f, rir, net, sig_len, mask=mask)
                    direct_path_signals.append(dir_path_signal)
                    full_rev_signals.append(full_rev_signal[:sig_len])
                    reconstructed_signals.append(rec_signal)
                else:
                    num_files += 1
        else:
            num_rirs += 1
    return reconstructed_signals, direct_path_signals, full_rev_signals


def create_22_channel_spect_set(directory, rir_directory, net, num_files=140, num_rirs=1, fs=16000, mask=False):
    rec_22_spects = []
    dir_path_22_spects = []
    full_rev_22_spects = []
    bin_weights = load_bin_weights()
    for i, rir_filename in enumerate(os.listdir(rir_directory)):
        if i == num_rirs:
            break
        r = os.path.join(rir_directory, rir_filename)
        if os.path.isfile(r):
            rir = ds.load_rir(r, fs)
            dir_rir = ds.get_direct_rir(rir, fs)
            for i, filename in enumerate(os.listdir(directory)):
                if i == num_files:
                    break
                f = os.path.join(directory, filename)
                if os.path.isfile(f):
                    dir_path_spect, _, _ = ds.create_spectrogram(f, dir_rir, norm=False)
                    sig_len = np.shape(dir_path_spect)[1]
                    dir_path_spect = pad_spect(dir_path_spect)
                    dir_path_spect_w = apply_bin_weights(dir_path_spect, bin_weights)
                    
                    input_spect, _, scl_vals = ds.create_spectrogram(f, rir, norm=True)
                    input_spect = input_spect[:, :sig_len]
                    full_rev_spect = rescale_spect(input_spect, scl_vals)
                    full_rev_spect = pad_spect(full_rev_spect)
                    full_rev_spect_w = apply_bin_weights(full_rev_spect, bin_weights)

                    rec_spect = apply_net_to_full_spect(net, input_spect, scl_vals, mask, rescale=True)
                    rec_spect = pad_spect(rec_spect)
                    rec_spect_w = apply_bin_weights(rec_spect, bin_weights)
                    
                    dir_path_22_spects.append(dir_path_spect_w)
                    full_rev_22_spects.append(full_rev_spect_w)
                    rec_22_spects.append(rec_spect_w)
                else:
                    num_files += 1
        else:
            num_rirs += 1
    return rec_22_spects, dir_path_22_spects, full_rev_22_spects
