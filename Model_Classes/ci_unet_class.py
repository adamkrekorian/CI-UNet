import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


class CI_Unet_64(nn.Module):
    def __init__(self):
        super(CI_Unet_64, self).__init__()
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

class CI_Unet_256(nn.Module):
    def __init__(self):
        super(CI_Unet_256, self).__init__()
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

def train(n_bins, net, train_data, test_data, mask=False, lr=0.01, reg=1e-3, epochs=10, loss_f=torch.nn.MSELoss()):
    # Model Training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    if device == 'cuda':
        print("Train on GPU...")
    else:
        print("Train on CPU...")
        
    # Initial learning rate
    INITIAL_LR = lr
    # Regularization
    REG = reg
    # Momentum
    MOMENTUM = 0.2
    # Total number of training epochs
    EPOCHS = epochs
    # Learning rate decay policy.
    DECAY_EPOCHS = 3
    DECAY = 0.7
    # Loss function
    criterion = loss_f
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(),
                             lr=INITIAL_LR, weight_decay=REG)
    # Initialize dataloaders
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)


    global_step = 0
    best_val_loss = 100
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
            for batch_idx, (inputs, targets) in enumerate(test_dataloader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                if mask:
                    outputs[outputs < 0.5] = 0
                    outputs[outputs >= 0.5] = 1
                loss = criterion(outputs, targets)
                val_loss += loss

        avg_loss = val_loss / len(test_dataloader)
        print("Validation loss: %.4f" % (avg_loss))

        if avg_loss < best_val_loss and i >= 10:
            print("Saving...")
            best_val_loss = avg_loss
            if mask:
                torch.save(net.state_dict(), "./Saved_Models/ci-unet-bm-%d.pt" % n_bins)
            else:
                torch.save(net.state_dict(), "./Saved_Models/ci-unet-%d.pt" % n_bins)

        # Handle the learning rate scheduler.
        if i % DECAY_EPOCHS == 0 and i != 0:
            current_learning_rate = current_learning_rate * DECAY
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_learning_rate

    print("Optimization finished.")
