import os
from os.path import dirname
import tensorflow as tf
import numpy as np
from neuralpixels.image import filters, make_collage
from scipy.misc import imread, imsave
from skimage.transform import resize
from collections import OrderedDict


def run_filters_test():
    print('neuralpixels.image.filters')
    num_passed = 0
    num_failed = 0

    def up_dir(num_up=1):
        x = dirname(__file__)
        for i in range(0, num_up):
            x = dirname(x)
        return x

    project_root = up_dir(4)
    test_img_path = os.path.join(project_root, 'assets', 'lenna.png')
    test_img = imread(test_img_path, mode='RGB')
    test_img = resize(test_img, [256, 256, 3], anti_aliasing=True, preserve_range=True).astype(np.float32)
    _test_img = tf.constant(test_img)
    test_img_expanded = np.expand_dims(test_img, axis=0)
    _test_img_expanded = tf.constant(test_img_expanded)

    tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    functions = [
        filters.gaus_blur,
        filters.box_blur,
        filters.sharpen,
        filters.edge,
        filters.sobel,
        filters.content_emphasis,
        filters.emboss
    ]
    collage_dict = OrderedDict()
    collage_dict['input'] = test_img.copy().astype(np.uint8)
    with tf.compat.v1.Session(config=tfconfig) as sess:
        func_name_width = 25
        for func in functions:
            name_str = '{} {} '.format(func.__name__, ''.rjust(func_name_width - len(func.__name__), '.'))
            print(' .{}'.format(name_str), end='')
            try:
                _out = func(_test_img_expanded)
                out = sess.run(_out)
                assert test_img_expanded.shape == out.shape, \
                    'Output shapes do not match. Output:{} target:{}'.format(out.shape, test_img_expanded.shape)

                print('PASSED')
                num_passed += 1
                collage_dict[str(func.__name__)] = np.squeeze(out, 0)
            except Exception as ex:
                print('FAIL')
                print('    {}'.format(str(ex)))
                num_failed += 1
    sess.close()
    if num_failed == 0:
        output_img_path = os.path.join(project_root, 'assets', 'python', 'tests', 'neuralpixels.image.filters.jpg')
        os.makedirs(dirname(output_img_path), exist_ok=True)
        imsave(output_img_path, make_collage(collage_dict))

    return num_passed, num_failed






