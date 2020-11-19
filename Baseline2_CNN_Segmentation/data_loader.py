import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from Baseline2_dataset import BasicDataset

# Some variable defintions
dir_img = 'overfit_data/'
dir_mask = 'overfit_masks/'
img_scale = 0.5
sqr_size = 250

# Obtain images and masks in 200x200 size format
dataset = BasicDataset(dir_img, dir_mask, img_scale)

# Extract Concatenated Images and masks (as well as images and reference images alone)
smallSet_data = np.zeros((10,6,sqr_size,sqr_size))
smallSet_img_alone = np.zeros((10,3,sqr_size,sqr_size))
smallSet_masks = np.zeros((10,1,sqr_size,sqr_size))
smallSet_ref_alone = np.zeros((10,3,sqr_size,sqr_size))
print(dataset[0]['ref_alone'].shape)

for i in range(len(dataset)):
    smallSet_data[i, :, :, :] = dataset[i]['image']
    smallSet_masks[i, :, :, :] = dataset[i]['mask']
    smallSet_img_alone[i, :, :, :] = dataset[i]['image_alone']
    smallSet_ref_alone[i, :, :, :] = dataset[i]['ref_alone']

# Show sample image
def show_sample_img(x):
    sample_img = torch.from_numpy(smallSet_img_alone[x])
    sample_img2 = torch.from_numpy(smallSet_masks[x])
    sample_img3 = torch.from_numpy(smallSet_ref_alone[x])
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(sample_img3.permute(1, 2, 0))
    axarr[1].imshow(sample_img.permute(1, 2, 0))
    axarr[2].imshow(sample_img2.permute(1, 2, 0))
    plt.show()
#show_sample_img(1)

# Divide into channels
img = smallSet_data
mask = smallSet_masks
channel0 = img[:,0,:,:]
channel1 = img[:,1,:,:]
channel2 = img[:,2,:,:]
channel3 = img[:, 3, :, :]
channel4 = img[:, 4, :, :]
channel5 = img[:, 5, :, :]
channel0_mask = mask[:, 0, :, :]

# Find Means
mean0 = channel0.mean()
mean1 = channel1.mean()
mean2 = channel2.mean()
mean3 = channel3.mean()
mean4 = channel4.mean()
mean5 = channel5.mean()
mean0_mask = channel0_mask.mean()

# Find STDs
std0 = channel0.std()
std1 = channel1.std()
std2 = channel2.std()
std3 = channel3.std()
std4 = channel4.std()
std5 = channel5.std()
std0_mask = channel0_mask.std()

def download_FCN():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True)
    model.eval()
    return model

def run_FCN(model, images):
    # sample execution (requires torchvision)
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)



'''
# Create transformation for images
transform_img = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((mean0, mean1, mean2, mean3, mean4, mean5),
                                                     (std0, std1, std2, std3, std4, std5))])
# Create transformation for mask images
transform_mask = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((mean0_mask), (std0_mask))])

smallSet_data = torchvision.datasets.ImageFolder(root = 'gdrive/My Drive/My_ASL_Images',
                                            transform=transform_img)
'''
'''
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(smallLoader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
'''