import tensorflow as tf
import numpy as np
from neuralpixels.layers.activations import prelu_clipped


def run_prelu_clipped_test():
    print('neuralpixels.activations.prelu_clipped')
    num_passed = 0
    num_failed = 0
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        _noise_wont_clip = tf.random_uniform([1, 10, 10, 3], minval=1, maxval=6, dtype=tf.float32)
        _noise_will_clip = tf.random_uniform([1, 10, 10, 3], minval=1, maxval=8, dtype=tf.float32)

        _output_wont_clip = prelu_clipped(_noise_wont_clip, name='prelu_clip1')
        _output_will_clip = prelu_clipped(_noise_will_clip, name='prelu_clip2')

        sess.run(tf.global_variables_initializer())

        noise_wont_clip, noise_will_clip, output_wont_clip, output_will_clip = sess.run(
            [_noise_wont_clip, _noise_will_clip, _output_wont_clip, _output_will_clip]
        )

        sess.close()
        acceptable_error = 2e-5
        func_name_width = 25
        try:
            name_str = '{} {} '.format('wont_clip', ''.rjust(func_name_width - len('wont_clip'), '.'))
            print(' .{}'.format(name_str), end='')
            wont_clip_diff = np.abs(noise_wont_clip - output_wont_clip)
            wont_clip_diff_max = np.max(wont_clip_diff)
            assert wont_clip_diff_max < acceptable_error, \
                'Outputs do not match for "wont" clip'
            num_passed += 1
            print('PASSED')
        except Exception as ex:
            print('FAIL')
            print('    {}'.format(str(ex)))
            num_failed += 1
        try:
            name_str = '{} {} '.format('will_clip', ''.rjust(func_name_width - len('will_clip'), '.'))
            print(' .{}'.format(name_str), end='')
            clipped_noise_will_clip = np.clip(noise_will_clip, -6.0, 6.0)
            will_clip_diff = np.abs(clipped_noise_will_clip - output_will_clip)
            will_clip_diff_max = np.max(will_clip_diff)
            assert will_clip_diff_max < acceptable_error, \
                'Outputs do not match for "will" clip'
            num_passed += 1
            print('PASSED')
        except Exception as ex:
            print('FAIL')
            print('    {}'.format(str(ex)))
            num_failed += 1
    return num_passed, num_failed
