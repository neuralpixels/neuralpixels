import os
import scipy.misc
import numpy as np


def get_img(src, dtype=None):
    img = scipy.misc.imread(src, mode='RGB')
    if dtype is not None:
        return img.astype(dtype)
    else:
        return img


def save_img(out_path, img):
    abs_out_path = os.path.abspath(out_path)
    dir_path = os.path.dirname(abs_out_path)
    os.makedirs(dir_path, exist_ok=True)
    img_to_save = np.clip(img.copy(), 0, 255).astype(np.uint8)
    scipy.misc.imsave(abs_out_path, img_to_save)
