from os.path import splitext
from os import listdir
import logging
import argparse
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img_trans = full_img.transpose((2, 0, 1))
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    img = torch.from_numpy(img_trans)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        print('GOT RESULT', probs.shape, type(probs))
        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor()])

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    scale = 0.5
    mask_threshold = 0.5
    in_files = [
        splitext(file)[0] for file in listdir('data/imgs')
        if not file.startswith('.')
    ]

    net = UNet(n_channels=6, n_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load('MODEL.pth', map_location=device))
    logging.info("Model loaded !")
    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img_file = glob(f'data/imgs/{fn}.*')
        ref_num = fn.split('_')[1]
        ref_file = glob(f'data/refs/01_{ref_num}.*')
        img = np.array(Image.open(img_file[0]).resize((200, 200)))
        ref = np.array(Image.open(ref_file[0]).resize((200, 200)))
        img = np.concatenate((img, ref), axis=2)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=scale,
                           out_threshold=mask_threshold,
                           device=device)

        result = mask_to_image(mask)
        result.save(f'predicted_masks/{fn}_MASK.jpg')
        print('SAVED IMAGE: ', fn + '_MASK.jpg')
