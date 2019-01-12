import tensorflow as tf
import numpy as np


def _convolve(inputs, kernel, pad=None, name=None):
    with tf.name_scope(name=name):
        if pad is not None:
            inputs_padded = tf.pad(inputs, [[0, 0], [pad, pad], [pad, pad], [0, 0]], "REFLECT")
        else:
            inputs_padded = inputs
        output = tf.nn.conv2d(
            inputs_padded,
            kernel,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        if pad is not None:
            output = tf.slice(output, [0, pad, pad, 0], tf.shape(inputs))
        return output


def one_to_three_channel_kernel(kernel):
    out = np.zeros([kernel.shape[0], kernel.shape[1], 3, 3], dtype=kernel.dtype)
    out[:, :, 0:1, 0:1] = kernel
    out[:, :, 1:2, 1:2] = kernel
    out[:, :, 2:3, 2:3] = kernel
    return out


def gaus_blur(inputs):
    a = np.zeros([3, 3, 1, 1])
    a[1, 1, :, :] = 0.25
    a[0, 1, :, :] = 0.125
    a[1, 0, :, :] = 0.125
    a[2, 1, :, :] = 0.125
    a[1, 2, :, :] = 0.125
    a[0, 0, :, :] = 0.0625
    a[0, 2, :, :] = 0.0625
    a[2, 0, :, :] = 0.0625
    a[2, 2, :, :] = 0.0625
    kernel = tf.constant(one_to_three_channel_kernel(a), dtype=tf.float32)
    return _convolve(inputs, kernel, pad=4, name='gaus_blur')


def box_blur(inputs, radius=4):
    diameter = radius * 2
    a = np.zeros([diameter, diameter, 1, 1])
    a[:, :, :, :] = 1 / (diameter * diameter)
    kernel = tf.constant(one_to_three_channel_kernel(a), dtype=tf.float32)

    return _convolve(inputs, kernel, pad=radius + 1, name='box_blur')


def sharpen(inputs, intensity=8, radius=3):
    radius = 3 if radius < 3 else radius
    a = np.zeros([radius, radius, 1, 1])
    total = radius * radius
    x = -intensity / (total - 1)
    center = radius // 2
    a[:, :, :, :] = x
    a[center, center, :, :] = intensity
    kernel = tf.constant(one_to_three_channel_kernel(a), dtype=tf.float32)
    return tf.add(inputs, _convolve(inputs, kernel, pad=radius + 1, name='sharpen'))


def edge(inputs, intensity=10, radius=3):
    radius = 3 if radius < 3 else radius
    a = np.zeros([radius, radius, 1, 1])
    total = radius * radius
    x = -intensity / (total - 1)
    center = radius // 2
    a[:, :, :, :] = x
    a[center, center, :, :] = intensity
    kernel = tf.constant(one_to_three_channel_kernel(a), dtype=tf.float32)
    return _convolve(inputs, kernel, pad=radius + 1, name='edge')


def sobel(inputs):
    a = np.zeros([3, 3, 1, 1])
    a[0, :, :, :] = 1
    a[0, 1, :, :] = 2
    a[2, :, :, :] = -1
    a[2, 1, :, :] = -2
    kernel = tf.constant(one_to_three_channel_kernel(a), dtype=tf.float32)
    return _convolve(inputs, kernel, pad=4, name='sobel')


def content_emphasis(inputs):
    a = np.zeros([3, 3, 1, 1])
    a[1, :, :, :] = -1
    a[:, 1, :, :] = -1
    a[1, 1, :, :] = 5
    kernel = tf.constant(one_to_three_channel_kernel(a), dtype=tf.float32)
    return _convolve(inputs, kernel, pad=4, name='content_emphasis')


def emboss(inputs):
    a = np.zeros([3, 3, 1, 1])
    a[0, 0, :, :] = -2
    a[0, 1, :, :] = -1
    a[1, 0, :, :] = -1
    a[1, 1, :, :] = 1
    a[1, 2, :, :] = 1
    a[2, 1, :, :] = 1
    a[2, 2, :, :] = 2
    kernel = tf.constant(one_to_three_channel_kernel(a), dtype=tf.float32)
    return _convolve(inputs, kernel, pad=4, name='emboss')
