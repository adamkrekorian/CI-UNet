import numpy as np
import matplotlib.pyplot as plt

import torch

from pystoi import stoi
from pesq import pesq

from Model_Classes.ci_unet_class import CI_Unet_64
from Data.dataset import extract_dataset
from Eval.reconstruct import recreate_from_spect_set

N_BINS = 64
fs = 16000

def plot_example_spects(testing_data):
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
    return [stoi_rec_avg, stoi_rev_avg, pesq_rec_avg, pesq_rev_avg]

def plot_intel_res(stoi_rec_avg, stoi_rev_avg, pesq_rec_avg, pesq_rev_avg):
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
    

    
if __name__=="__main__":
    # Load Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device = {}".format(device))
    net = CI_Unet_64()
    net = net.to(device)

    net.load_state_dict(torch.load("./Saved_Models/ci-unet-64.pt", map_location=torch.device(device)))
    net.eval()
    
    # Data paths
    directory_test = "./Data/Speech_Files/Testing/"
    rir_directory_test = "./Data/RIR_Files/Testing/"

    # Extract Dataset
    test_dataset = extract_dataset(directory_test, rir_directory_test,
                                   140, 1,
                                   training_set=False, mask=False)

    # Plot Example Spect Comparison
    plot_example_spects(test_dataset)

    intel_res = compute_intel_metrics(directory_test, rir_directory_test, net, mask=True)
    plot_intel_res(intel_res[0], intel_res[1], intel_res[2], intel_res[3])

    



    
