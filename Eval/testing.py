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
import Eval.reconstruct as rc

N_BINS = 64
FS = 16000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_norm_comp(fig, ax, spect, title):
    temp_im = ax.imshow(spect, cmap=plt.get_cmap("jet"))
    ax.set_xlabel("Frames")
    ax.set_ylabel("Channels")
    ax.set_aspect('auto')
    fig.colorbar(temp_im, ax=ax)
    ax.set_title(title)


def test_norm(file, rir, net, ind):
    dir_rir = ds.get_direct_rir(rir)
    spect, _, _ = ds.create_spectrogram(file, rir, norm=False)
    norm_spect, min_, max_ = ds.normalize(spect) 
    dir_spect, _, _ = ds.create_spectrogram(file, dir_rir, norm=False)
    net_spect = rc.apply_net_to_full_spect(net, norm_spect, [min_, max_], rescale=True)

    spects = [np.log10(spect), np.log10(net_spect), np.log10(dir_spect)]
    titles = ["Original Spectrogram", "Net Spectrogram", "Target Spectrogram"]

    fig, axs = plt.subplots(len(spects), 1, figsize=(10, 12), dpi=200)    
    
    for i, ax in enumerate(axs):
        plot_norm_comp(fig, ax, spects[i], titles[i]) 

    plt.savefig(f"./Eval/Results/Comp/comp_{ind}", dpi=200, bbox_inches="tight")
    
def norm_comp_set(directory, rir_directory, net, num_files=140, num_rirs=1):
    for i, rir_filename in enumerate(os.listdir(rir_directory)):
        if i == num_rirs:
            break
        r = os.path.join(rir_directory, rir_filename)
        if os.path.isfile(r):
            rir = ds.load_rir(r, FS)
            for i, filename in enumerate(os.listdir(directory)):
                if i == num_files:
                    break
                f = os.path.join(directory, filename)
                if os.path.isfile(f):
                    test_norm(f, rir, net, i)
                else:
                    num_files += 1
