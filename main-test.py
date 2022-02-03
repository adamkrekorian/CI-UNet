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
from Eval.testing import norm_comp_set

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
        plt.savefig("./Eval/Results/res-eval-64-bm-%d.png" % i)
    else:
        plt.savefig("./Eval/Results/res-eval-64-%d.png" % i)
    


def compute_intel_metrics(directory, rir_directory, net, mask=False):
    reconstructed_signals, direct_path_signals, full_rev_signals = recreate_from_spect_set(directory, rir_directory, net, num_files=140, num_rirs=1, mask=mask)

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

            # Save .wav file of speech
            wavfile.write("./Eval/Results/full_reverb_speech.wav", fs, full_rev)

            plt.subplot(3, 1, 2)
            plt.plot(rec)
            plt.xlabel("Samples")
            plt.ylabel("Normalized Amplitude")
            if mask:
                plt.title("Recreated Signal (IBM)")
                
                # Save .wav file of speech
                wavfile.write("./Eval/Results/rec_mask_speech.wav", fs, rec)
                
            else:
                plt.title("Recreated Signal")
                
                # Save .wav file of speech
                wavfile.write("./Eval/Results/rec_speech.wav", fs, rec)
                
            plt.subplot(3, 1, 3)
            plt.plot(dir_path)
            plt.xlabel("Samples")
            plt.ylabel("Normalized Amplitude")
            plt.title("Direct Path Signal")

            # Save .wav file of speech
            wavfile.write("./Eval/Results/dir_path_speech.wav", fs, dir_path)

            if mask:
                plt.savefig("./Eval/Results/res-eval-64-rec-comp-bm-%d.png" % i)
            else:
                plt.savefig("./Eval/Results/res-eval-64-rec-comp-%d.png" % i)

                
        stoi_rec_sum += stoi(dir_path, rec, fs)
        stoi_rev_sum += stoi(dir_path, full_rev, fs)
            
        pesq_rec_sum += pesq(fs, dir_path, rec, 'wb')
        pesq_rev_sum += pesq(fs, dir_path, full_rev, 'wb')
    
    stoi_rec_avg = stoi_rec_sum / len(reconstructed_signals)
    stoi_rev_avg = stoi_rev_sum / len(reconstructed_signals)

    pesq_rec_avg = pesq_rec_sum / len(reconstructed_signals)
    pesq_rev_avg = pesq_rev_sum / len(reconstructed_signals)
    return [stoi_rec_avg, stoi_rev_avg, pesq_rec_avg, pesq_rev_avg]

def plot_intel_res(stoi_rec_avg, stoi_rev_avg, pesq_rec_avg, pesq_rev_avg, mask=False):
    if mask:
        labels = ["Reverberant Signal", "Recreated Signal (IBM)"]
    else:
        labels = ["Reverberant Signal", "Recreated Signal"]
    colors = ["red", "blue"]
    stoi_values = [stoi_rev_avg, stoi_rec_avg]
    pesq_values = [pesq_rev_avg, pesq_rec_avg]
    print(stoi_values)
    print(pesq_values)
    
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    
    rect0 = axs[0].bar(labels, stoi_values, color=colors)
    axs[0].set_xlabel("Signal")
    axs[0].set_ylabel("STOI Value")
    axs[0].set_title("Mean STOI results (N = 140)")
    axs[0].set_ylim((0, 1))
    for i, v in enumerate(stoi_values):
        xloc = rect0[i].get_x() + rect0[i].get_width() / 2
        yloc = 1.05 * rect0[i].get_height()
        axs[0].text(xloc, yloc, f"{v:.3f}")
    
    rect1 = axs[1].bar(labels, pesq_values, color=colors)
    axs[1].set_xlabel("Signal")
    axs[1].set_ylabel("PESQ Value")
    axs[1].set_title("Mean PESQ results (N = 140)")
    axs[1].set_ylim((0.5, 1.5))
    for i, v in enumerate(pesq_values):
        xloc = rect1[i].get_x() + rect1[i].get_width() / 2
        yloc = 1.05 * rect1[i].get_height()
        axs[1].text(xloc, yloc, f"{v:.3f}")
    
    
    if mask:
        plt.savefig("./Eval/Results/intel-eval-bm-64.png")
    else:
        plt.savefig("./Eval/Results/intel-eval-64.png")


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
                #temp_im = ax.imshow(temp_sp, vmin=0, vmax=np.max(temp_sp), cmap=plt.get_cmap("jet"))
                temp_im = ax.imshow(temp_sp, cmap=plt.get_cmap("jet"))
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
            #vocode_rec = eng.vocodeCIStimulus(rec_spect_ml)
            #vocode_rev = eng.vocodeCIStimulus(full_spect_ml)

        if np.isnan(ecm_rec_val):
            num_nan += 1
            
        ecm_rec_sum += np.nan_to_num(ecm_rec_val)
        ecm_rev_sum += np.nan_to_num(ecm_rev_val)

    ecm_rec_avg = ecm_rec_sum / (len(rec_spects) - num_nan)
    ecm_rev_avg = ecm_rev_sum / (len(rec_spects) - num_nan)
    return [ecm_rec_avg, ecm_rev_avg]
    
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

    norm_comp_set(directory_test, rir_directory_test, net, num_files=20)
