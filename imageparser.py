from glob import glob
from PIL import Image
from os import listdir
from os.path import splitext
import numpy as np

ids = [
    splitext(file)[0] for file in listdir('data/imgs')
    if not file.startswith('.')
]

for i in ids:
    mask_file = glob(f'data/masks/{i}.*')
    print('GOT MASK FILE', mask_file, i)
    mask = Image.open(mask_file[0]).convert('L')
    mask = mask.resize((200, 200))
    mask = mask.point(lambda x: 0 if x < 128 else 255, '1')
    print('GOT IMG', np.array(mask).shape)
    mask.save(f'data/parsed_masks/{i}.jpg')
