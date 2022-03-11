import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import wavfile
import torch

from pystoi import stoi
from pesq import pesq

import matlab
import matlab.engine

from Model_Classes.ci_unet_class import CI_Unet_64
from Data.dataset import extract_dataset
from Eval.reconstruct import recreate_from_spect_set, create_22_channel_spect_set

N_BINS = 64
fs = 16000

def plot_example_spects(testing_data, mask=False):
    i = 1
    feature, label = testing_data.__getitem__(i)
    feature = torch.unsqueeze(feature, 0)
    with torch.no_grad():
        feature = feature.to(device)
        output = net(feature)
        if mask:
            output[output < 0.5] = 0
            output[output >= 0.5] = 1
            output = feature * output

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
    if mask:
        plt.title("Spectrogram with Applied Binary Mask (64 Bins)")
    else:
        plt.title("Predicted Spectrogram (64 Bins)")
    plt.xlabel("Time Domain")
    plt.ylabel("Frequency Domain")

    plt.subplot(133)
    plt.imshow(np.flipud(label[0, :, :].cpu()), vmin=-1, vmax=1, cmap=plt.get_cmap("jet"))
    plt.colorbar()
    if mask:
        plt.title("Ideal Binary Mask (64 Bins)")
    else:
        plt.title("Direct Path Spectrogram (64 Bins)")
    plt.xlabel("Time Domain")
    plt.ylabel("Frequency Domain")
    if mask:
        fig_filename = f"./Eval/Results/res-eval-64-bm-{i}.png"
    else:
        fig_filename = f"./Eval/Results/res-eval-64-{i}.png"

    plt.savefig(fig_filename, dpi=200, bbox_inches="tight")


def compute_intel_metrics(directory, rir_directory, net, mask=False):
    reconstructed_signals, direct_path_signals, full_rev_signals = recreate_from_spect_set(directory, rir_directory, net, num_files=140, num_rirs=1, mask=mask)

    stoi_rec_sum = 0
    stoi_rev_sum = 0

    pesq_rec_sum = 0
    pesq_rev_sum = 0

    for i in range(len(reconstructed_signals)):
        dir_path = np.real(direct_path_signals[i])
        rec = np.real(reconstructed_signals[i])
        full_rev = np.real(full_rev_signals[i])
    
        dir_path = dir_path/np.max(dir_path)
        rec = rec/np.max(rec)
        full_rev = full_rev/np.max(full_rev)
        if i in range(10):
            if not os.path.exists(f"./Eval/Results/Speech/sentence_{i}"):
                os.makedirs(f"./Eval/Results/Speech/sentence_{i}")
            
            plt.figure(figsize=(15, 15))
        
            plt.subplot(3, 1, 1)
            plt.plot(full_rev)
            plt.xlabel("Samples")
            plt.ylabel("Normalized Amplitude")
            plt.title("Reverberant Signal")

            # Save .wav file of speech
            wavfile.write(f"./Eval/Results/Speech/sentence_{i}/full_reverb_speech.wav", fs, full_rev)

            plt.subplot(3, 1, 2)
            plt.plot(rec)
            plt.xlabel("Samples")
            plt.ylabel("Normalized Amplitude")

            if mask:
                plt.title("Recreated Signal (IBM)")
                
                # Save .wav file of speech
                wavfile.write(f"./Eval/Results/Speech/sentence_{i}/rec_mask_speech.wav", fs, rec)
                
            else:
                plt.title("Recreated Signal")
                
                # Save .wav file of speech
                wavfile.write(f"./Eval/Results/Speech/sentence_{i}/rec_speech.wav", fs, rec)
                
            plt.subplot(3, 1, 3)
            plt.plot(dir_path)
            plt.xlabel("Samples")
            plt.ylabel("Normalized Amplitude")
            plt.title("Direct Path Signal")

            # Save .wav file of speech
            wavfile.write(f"./Eval/Results/Speech/sentence_{i}/dir_path_speech.wav", fs, dir_path)

            if mask:
                fig_filename = f"./Eval/Results/Speech/sentence_{i}/res-eval-64-rec-comp-bm-{i}.png"
            else:
                fig_filename = f"./Eval/Results/Speech/sentence_{i}/res-eval-64-rec-comp-{i}.png"

            plt.savefig(fig_filename, dpi=200, bbox_inches="tight")
                
        stoi_rec_sum += stoi(dir_path, rec, fs)
        stoi_rev_sum += stoi(dir_path, full_rev, fs)
            
        pesq_rec_sum += pesq(fs, dir_path, rec, 'wb')
        pesq_rev_sum += pesq(fs, dir_path, full_rev, 'wb')
    
    stoi_rec_avg = stoi_rec_sum / len(reconstructed_signals)
    stoi_rev_avg = stoi_rev_sum / len(reconstructed_signals)

    pesq_rec_avg = pesq_rec_sum / len(reconstructed_signals)
    pesq_rev_avg = pesq_rev_sum / len(reconstructed_signals)
    return [stoi_rev_avg, stoi_rec_avg, pesq_rev_avg, pesq_rec_avg]

def plot_intel_res(avgs, mask=False):
    if mask:
        labels = ["Reverberant Signal", "Recreated Signal (IBM)"]
    else:
        labels = ["Reverberant Signal", "Recreated Signal"]
    colors = ["red", "blue"]
    stoi_values = avgs[:2]#[stoi_rev_avg, stoi_rec_avg]
    pesq_values = avgs[2:4] #[pesq_rev_avg, pesq_rec_avg]
    ecm_values = avgs[4:]

    names = ["STOI", "PESQ", "ECM"]
    print(pesq_values)

    def plot_metric(ax, name, vals):
        rect = ax.bar(labels, vals, color=colors)
        ax.set_xlabel("Signal")
        ax.set_ylabel(f"{name} Value")
        ax.set_title(f"Mean {name} results (N = 140)")
        ax.set_ylim((0.5, 1.6))
        for i, v in enumerate(vals):
            xloc = rect[i].get_x() + rect[i].get_width() / 2
            yloc = 1.05 * rect[i].get_height()
            ax.text(xloc, yloc, f"{v:.3f}")
    
    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    for i, ax in enumerate(axs):
        plot_metric(ax, names[i], avgs[2*i:2*(i+1)]) 
        
    if mask:
        fig_filename = "./Eval/Results/intel-eval-bm-64.png"
    else:
        fig_filename = "./Eval/Results/intel-eval-64.png"

    plt.savefig(fig_filename, dpi=200, bbox_inches="tight")

def compute_ecm_metrics(directory, rir_directory, net, mask=False):
    rec_spects, dir_spects, full_spects = create_22_channel_spect_set(directory, rir_directory, net, mask=mask)
    ecm_rec_sum = 0
    ecm_rev_sum = 0

    num_nan = 0
    
    eng = matlab.engine.start_matlab()
    
    for i in range(len(rec_spects)):
        dir_spect = dir_spects[i]
        rec_spect = rec_spects[i]
        full_spect = full_spects[i]
    
        dir_spect /= np.max(dir_spect)
        rec_spect /= np.max(rec_spect)
        full_spect /= np.max(full_spect)

        if i == 0:
            temp_spects = [full_spect, rec_spect, dir_spect]
            fig, axs = plt.subplots(3, 1, figsize=(20,20))
        
            for ax, temp_sp in zip(axs, temp_spects):
                temp_im = ax.imshow(np.log10(temp_sp), cmap=plt.get_cmap("jet"))
                ax.set_xlabel("Frames")
                ax.set_ylabel("Channels")
                ax.set_aspect('auto')
                fig.colorbar(temp_im, ax=ax)
            
            axs[0].set_title("Reverberant Spectrogram")
            
            if mask:
                axs[1].set_title("Recreated Spectrogram (IBM)")                
            else:
                axs[1].set_title("Recreated Spectrogram")
        
            axs[2].set_title("Direct Path Spectrogram")

            
            if mask:
                fig_filename = "./Eval/Results/res-eval-64-rec-spect-comp-bm-%d.png" % i
            else:
                fig_filename = "./Eval/Results/res-eval-64-rec-spect-comp-%d.png" % i

            plt.savefig(fig_filename, dpi=200, bbox_inches="tight")

        dir_spect_ml = matlab.double(dir_spect.tolist())
        rec_spect_ml = matlab.double(rec_spect.tolist())
        full_spect_ml = matlab.double(full_spect.tolist())

        fs_ml = matlab.double([500])

        ecm_rec_val = eng.calculateEcm(dir_spect_ml, rec_spect_ml, fs_ml, 'mean')
        ecm_rev_val = eng.calculateEcm(dir_spect_ml, full_spect_ml, fs_ml, 'mean')

        if i == 0:
            np.savetxt('rec_spect.csv', rec_spect, delimiter=',')
            np.savetxt('full_spect.csv', full_spect, delimiter=',')

        if np.isnan(ecm_rec_val):
            num_nan += 1
            
        ecm_rec_sum += np.nan_to_num(ecm_rec_val)
        ecm_rev_sum += np.nan_to_num(ecm_rev_val)

    ecm_rec_avg = ecm_rec_sum / (len(rec_spects) - num_nan)
    ecm_rev_avg = ecm_rev_sum / (len(rec_spects) - num_nan)
    return [ecm_rev_avg, ecm_rec_avg]
    
if __name__=="__main__":
    masking = False
    
    # Load Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device = {}".format(device))
    net = CI_Unet_64()
    net = net.to(device)
    if masking:
        net.load_state_dict(torch.load("./Saved_Models/ci-unet-bm-64.pt", map_location=torch.device(device)))
    else:
        net.load_state_dict(torch.load("./Saved_Models/ci-unet-64.pt", map_location=torch.device(device)))
    net.eval()
    
    # Data paths
    directory_test = "./Data/Speech_Files/Testing/"
    rir_directory_test = "./Data/RIR_Files/Testing/"

    # Extract Dataset
    print("Extracting testing set...")
    test_dataset = extract_dataset(directory_test, rir_directory_test,
                                   140, 1,
                                   training_set=False, mask=masking)

    # Plot Example Spect Comparison
    plot_example_spects(test_dataset, mask=masking)

    intel_res = compute_intel_metrics(directory_test, rir_directory_test, net, mask=masking)
    
    print("Computing ECM Metrics...")
    ecm_res = compute_ecm_metrics(directory_test, rir_directory_test, net, mask=masking)
    plot_intel_res(intel_res + ecm_res,  mask=masking)
    print(ecm_res)
