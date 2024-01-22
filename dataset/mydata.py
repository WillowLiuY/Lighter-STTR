import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from natsort import natsorted
import numpy as np

class ScaredDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        data_dir: Directory for the Scared Dataset
        split: train, validate, or test        
        """
        super(ScaredDataset, self).__init__()
        self.data_dir = data_dir
        self.split = split

        # Read Data
        left_dir = os.path.join(self.data_dir, 'img_left') # Dir containing left images
        self.left_img = [os.path.join(left_dir, img) for img in os.listdir(left_dir)]
        self.left_img = natsorted(self.left_img)

        # Augmentation
        self.transform = None

    def __len__(self):
        return len(self.left_img)
    
    def __getitem__(self, idx):
        result = {}
        left_fname = self.left_img[idx]
        result['left'] = np.array(Image.open(left_fname)).astype(np.uint8)

        right_fname = left_fname.replace('img_left', 'img_right')
        result['right'] = np.array(Image.open(right_fname)).astype(np.uint8)

        disp_fname = left_fname.replace('img_left', 'disp_left').replace('.png', 'pfm')
        disp, _ = readPFM(disp_fname)
        result['disp'] = disp # Disparity map

        occ_fname = left_fname.replace('img_left', 'occ_left')
        result['occ_mask'] = np.array(Image.open(occ_fname)).astype(np.uint8) == 128
        result['disp'][result['occ_mask']] == 0.0 # Apply occlusion mask to disparity map

        result = augment(result, self.transform)
        return result