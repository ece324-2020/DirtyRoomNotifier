# Borrowed from ECE324 Lecture 27

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from PIL import Image
import time

from Autoencoder_Class import Autoencoder
from Baseline_Dataset import Load_Set

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load small data set for over-fitting
start = time.time()
print('Begin Loading Data...')
smallSetImg, smallSetMsk = Load_Set(overfit_set = False)
end = time.time()
time2 = end - start
print('Done Loading Data')
print("Time Taken to Load Data: " + str(time2))

def train(model, num_epochs=5, batch_size=1, learning_rate=0.001):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5) # <--

    img_loader = torch.utils.data.DataLoader(smallSetImg, batch_size=batch_size, shuffle=False)
    msk_loader = torch.utils.data.DataLoader(smallSetMsk, batch_size=batch_size, shuffle=False)

    outputs = []
    full_loss = []
    epoch_tracker = []

    # Keep track of time it takes to run
    start = time.time()

    for epoch in range(num_epochs):
        for img, mask in zip(img_loader, msk_loader):
            img, mask = img.to(device), mask.to(device)
            recon = model(img,batch_size)
            loss = criterion(recon, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        full_loss.append(float(loss))
        epoch_tracker.append(epoch)
        outputs.append((epoch, img, recon),)   # store the outputs for viewing below

    end = time.time()
    time1 = end - start
    print("Run Time: " + str(time1))

    return outputs, full_loss, epoch_tracker

# Run Training Loop
net = Autoencoder()
net.to(device)

max_epochs = 500
outputs, loss, epochs = train(net, num_epochs=max_epochs)

# Plot accuracy vs step
plt.plot(epochs, loss)
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

def Segment_Evolution():
    # See evolution of images
    for k in range(0, max_epochs, 100):
        plt.figure(figsize=(9, 2))
        imgs = outputs[k][1].cpu().detach().numpy()
        recon = outputs[k][2].detach().cpu().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i + 1)
            plt.imshow(item[0])

        for i, item in enumerate(recon):
            if i >= 9: break
            plt.subplot(2, 9, 9 + i + 1)
            plt.imshow(item[0])
    plt.show()

#Segment_Evolution()

imgs = outputs[max_epochs-1][1].cpu().detach().numpy()
recons = outputs[max_epochs-1][2].cpu().detach().numpy()
plt.subplot(1, 2, 1)
plt.imshow(imgs[0][0])
plt.subplot(1, 2, 2)
plt.imshow(recons[0][0])
plt.show()