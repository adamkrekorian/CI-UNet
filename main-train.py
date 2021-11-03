import torch
from torch.utils.data import DataLoader

from Model_Classes.ci_unet_class import CI_Unet_64, train
from Data.dataset import extract_dataset

N_BINS = 64
fs = 16000

if __name__=="__main__":
    # Load Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device = {}".format(device))
    
    # Data paths
    directory_train = "./Data/Speech_Files/Training/"
    rir_directory_train = "./Data/RIR_Files/24Training/"

    directory_test = "./Data/Speech_Files/Testing/"
    rir_directory_test = "./Data/RIR_Files/Testing/"

    # Extract Dataset
    train_dataset = extract_dataset(directory_train, rir_directory_train, num_files=100, num_rirs=24)
    test_dataset = extract_dataset(directory_test, rir_directory_test, num_files=140, num_rirs=1)

    # Train
    train(N_BINS, CI_Unet_64(), train_dataset, test_dataset)

    

    



    
