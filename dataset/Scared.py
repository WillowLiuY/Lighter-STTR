import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from natsort import natsorted
import numpy as np
from dataset.preprocess import custom_transform
from utilities.python_pfm import readPFM

class ScaredDataset(Dataset):
    def __init__(self, data_path, split='train'):
        """
        data_path: Directory for the Scared Dataset
        split: train, val, or test        
        """
        super(ScaredDataset, self).__init__()
        self.data_path = data_path
        self.split = split

        # Load images from left image directory
        left_path = os.path.join(self.data_path, 'img_left')
        self.left_img = [os.path.join(left_path, file) for file in os.listdir(left_path)]
        self.left_img = natsorted(self.left_img)

        # transformations
        self.transform = None

    def __len__(self):
        return len(self.left_img)
    
    def __getitem__(self, idx):
        inputs = {}
        left_img_path = self.left_img[idx]
        data['left'] = np.array(Image.open(left_img_path)).astype(np.uint8)

        right_img_path = left_img_path.replace('img_left', 'img_right')
        data['right'] = np.array(Image.open(right_img_path)).astype(np.uint8)

        # disparity maps
        disp_path = left_img_path.replace('img_left', 'disp_left').replace('.png', 'pfm')
        disparity, _ = readPFM(disp_path)
        data['disp'] = disparity

        # occlution masks
        occ_path = left_img_path.replace('img_left', 'occ_left')
        data['occ_mask'] = np.array(Image.open(occ_path)).astype(np.uint8) == 128
        data['disp'][data['occ_mask']] == 0.0

        data = custom_transform(data, self.transform)
        return data