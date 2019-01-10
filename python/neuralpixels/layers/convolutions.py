import tensorflow as tf
import numpy as np
import itertools


def subpixel_upscale(inputs, zoom=2, name='subpixel', trainable=False):
    """Subpixel Upscaling

    Source for original implementation: https://arxiv.org/abs/1609.05158

    Learn more about this implementation at:
    https://neuralpixels.com/subpixel-upscaling/

    :param inputs: A `Tensor`
    :param zoom: The amount to zoom
    :param name: Variable scope name
    :param trainable: If True, the original subpixel kernel will be used as an initializer and trained from there.
    :return:
    """
    with tf.variable_scope(name):
        r = zoom
        batch_size, rows, cols, in_channels = inputs.get_shape().as_list()
        kernel_filter_size = r
        out_channels = int(in_channels // (r * r))

        kernel_shape = [kernel_filter_size, kernel_filter_size, out_channels, in_channels]
        kernel = np.zeros(kernel_shape, np.float32)

        # Build the kernel so that a 4 pixel cluster has each pixel come from a separate channel.
        for c in range(0, out_channels):
            i = 0
            for x, y in itertools.product(range(r), repeat=2):
                kernel[y, x, c, c * r * r + i] = 1
                i += 1

        new_rows, new_cols = int(rows * r), int(cols * r)
        new_shape = [batch_size, new_rows, new_cols, out_channels]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1, r, r, 1]

        if trainable:
            kernel_weights = tf.get_variable(
                name='kernel',
                shape=kernel.shape,
                initializer=tf.constant_initializer(kernel)
            )
        else:
            kernel_weights = kernel

        out = tf.nn.conv2d_transpose(inputs, kernel_weights, tf_shape, strides_shape, padding='VALID')

        return out
