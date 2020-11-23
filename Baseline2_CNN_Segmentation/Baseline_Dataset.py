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

def Load_Set(overfit_set = False):
    # Some variable defintions
    if overfit_set:
        dir_img = 'overfit_data/'
        dir_mask = 'overfit_masks/'
    else:
        dir_img = 'data/imgs/'
        dir_mask = 'data/masks/'

    images = torchvision.datasets.ImageFolder(root = dir_img, transform=transforms.ToTensor())
    masks = torchvision.datasets.ImageFolder(root = dir_mask, transform=transforms.ToTensor())

    # resizing our images to desired size
    size = 100
    images_resized = []
    masks_resized = []

    for i in range(len(images)):
        # resize images
        temp_img = transforms.ToPILImage()(images[i][0]).convert("RGB").resize((size, size))
        temp_mask = transforms.ToPILImage()(masks[i][0]).convert("RGB").resize((size, size))

        # Make Images Grayscale (1 channel)
        temp_img = transforms.Grayscale()(temp_img)
        temp_mask = transforms.Grayscale()(temp_mask)

        # Transform images to tensor form from PIL form
        pil_to_tensor = transforms.ToTensor()(temp_img)
        pil_to_tensor_mask = transforms.ToTensor()(temp_mask)

        # Store images
        images_resized.append(pil_to_tensor)
        masks_resized.append(pil_to_tensor_mask)

    # Stack all resized tensors
    images_resized = torch.stack(images_resized)
    masks_resized = torch.stack(masks_resized)

    # Final Dataset
    smallSetImg = images_resized
    smallSetMsk = masks_resized

    return smallSetImg, smallSetMsk
