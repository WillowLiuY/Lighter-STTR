import numpy as np
import torch
from albumentations import Compose
from dataset.stereo_albumentation import Normalize, ToTensor

# use ImageNet stats for normalization
__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

normalization = Compose([Normalize(always_apply=True),
                         ToTensor(always_apply=True)], p=1.0)


def denormalize(img):
    """
    De-normalize a tensor back to original image

    :param img: normalized image, [C,H,W]
    :return: original image, [H,W,C]
    """
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0)  # Conver to H,W,C
        img *= torch.tensor(__imagenet_stats['std'])
        img += torch.tensor(__imagenet_stats['mean'])
        return img.numpy()
    else:
        img = img.transpose(1, 2, 0)
        img *= np.array(__imagenet_stats['std'])
        img += np.array(__imagenet_stats['mean'])
        return img


def get_left_occlusion(w, disp):
    """
    Compute occluded region on the left image border

    :param w: image width
    :param disp: disparity of right image
    :return: occlusion mask
    """
    x_coords = np.arange(0, w)[None, :]  # 1xW
    shifted_x_coords = x_coords - disp
    left_occ_mask = shifted_x_coords < 0  # True where occluded

    return left_occ_mask


def get_right_occlusion(w, disp):
    """
    Compute occluded region on the right image border
    """
    x_coords = np.arange(0, w)[None, :]  # 1xW
    shifted_x_coords = x_coords + disp
    right_occ_mask = shifted_x_coords > w 

    return right_occ_mask

def custom_transform(inputs, transformation):
    """
    apply custom augmentation and handle occlusions.

    :param inputs: Inputs dictionary with images and disparity maps
    :return: dictionary with transformed images and updated occlusion masks
    """

    if transformation:
        inputs = transformation(**inputs)

    w = inputs['disp'].shape[-1]
    # clamp disparity values to be within [0, width]
    inputs['disp'] = np.clip(inputs['disp'], 0, w)

    # compute occlusion for the left image
    left_occlusion = get_left_occlusion(w, inputs['disp'])
    inputs['occ_mask'][left_occlusion] = True  # update
    inputs['occ_mask'] = np.ascontiguousarray(inputs['occ_mask'])

    # compute occlusion for the right image
    try:
        right_occlusion = get_right_occlusion(w, inputs['disp_right'])
        inputs['occ_mask_right'][right_occlusion] = 1
        inputs['occ_mask_right'] = np.ascontiguousarray(inputs['occ_mask_right'])
    except KeyError:
        # print('No disp mask right, check if dataset is KITTI')
        inputs['occ_mask_right'] = np.zeros_like(occ_mask, dtype=np.bool_)

    # clean up disparity map
    inputs.pop('disp_right', None)

    # set occluded disparity areas to 0
    inputs['disp'][inputs['occ_mask']] = 0
    inputs['disp'] = np.ascontiguousarray(inputs['disp'], dtype=np.float32)

    # return normalized image
    return normalization(**inputs)
