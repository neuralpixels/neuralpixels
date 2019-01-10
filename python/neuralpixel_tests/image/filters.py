import os
from os.path import dirname
import tensorflow as tf
import numpy as np
from neuralpixels.image import filters
from scipy.misc import imread


def run_filters_test():
    print('neuralpixels.image.filters')
    num_passed = 0
    num_failed = 0

    project_root = dirname(dirname(dirname(dirname(__file__))))
    test_img_path = os.path.join(project_root, 'assets', 'lenna.png')
    test_img = imread(test_img_path, mode='RGB').astype(np.float32)
    _test_img = tf.constant(test_img)
    test_img_expanded = np.expand_dims(test_img, axis=0)
    _test_img_expanded = tf.constant(test_img_expanded)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
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

    with tf.Session(config=tfconfig) as sess:
        func_name_width = 25
        for func in functions:
            name_str = '{} {} '.format(func.__name__, ''.rjust(func_name_width - len(func.__name__), '.'))
            print(' .{}'.format(name_str), end='')
            try:
                _out = filters.gaus_blur(_test_img_expanded)
                out = sess.run(_out)
                assert test_img_expanded.shape == out.shape, \
                    'Output shapes do not match. Output:{} target:{}'.format(out.shape, test_img_expanded.shape)

                print('PASSED')
                num_passed += 1
            except Exception as ex:
                print('FAIL')
                print('    {}'.format(str(ex)))
                num_failed += 1
    sess.close()
    return num_passed, num_failed






