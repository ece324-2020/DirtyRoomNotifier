from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

empty_mask = Image.new("L", (200, 200), (255))


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.refs_dir = 'data/refs/'
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [
            splitext(file)[0] for file in listdir(imgs_dir)
            if not file.startswith('.')
        ]
        self.ref_ids = [
            splitext(file)[0] for file in listdir(self.refs_dir)
            if not file.startswith('.')
        ]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids) + len(self.ref_ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((200, 200))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        if i >= len(self.ids):
            idx = self.ref_ids[i - len(self.ids) - 1]
            idref = idx.split('_')[1]  # i.e. A1
            img_file = glob(self.refs_dir + '01_' + idref + '.*')
            ref_file = glob(self.refs_dir + idx + '.*')
            ref = Image.open(ref_file[0])
            img = Image.open(img_file[0])
            mask = empty_mask
        else:
            idx = self.ids[i]
            idref = idx.split('_')[1]  # i.e. A1
            mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
            ref_file = glob(self.refs_dir + '01_' + idref + '.*')
            img_file = glob(self.imgs_dir + idx + '.*')

            assert len(ref_file) >= 1, \
                f'Reference image not found for the ID {idx}: {mask_file}'
            assert len(mask_file) >= 1, \
                f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
            assert len(img_file) >= 1, \
                f'Either no image or multiple images found for the ID {idx}: {img_file}'
            mask = Image.open(mask_file[0])
            ref = Image.open(ref_file[0])
            img = Image.open(img_file[0])

        ref = self.preprocess(ref, self.scale)
        img = self.preprocess(img, self.scale)
        img = np.concatenate((img, ref))
        mask = self.preprocess(mask, self.scale)
        print('GOT mask, ref SIZES', img.shape, mask.shape)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
