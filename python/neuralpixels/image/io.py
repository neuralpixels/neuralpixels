import os
import scipy.misc
import numpy as np
from glob import glob


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


def get_img_paths(folder):
    img_paths = []
    img_types = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
    glob_pre = folder if folder.endswith('/') else folder + '/'
    for img_type in img_types:
        paths = glob('{}**/*.{}'.format(glob_pre, img_type), recursive=True)
        img_paths += paths
    return img_paths