# -*- coding: utf-8 -*-
"""Neural Pixels Convolutions

A variety of convolutional layers.

Todo:
    * Add exceptions

"""
import tensorflow as tf
import numpy as np
import itertools


def subpixel_upscale(inputs, zoom=2, name='subpixel_upscale', trainable=False, legacy=False, weights=None):
    """Subpixel Upscale

    Source for original implementation: https://arxiv.org/abs/1609.05158

    If the legacy flag is passed, the kernel generated will produce an output
    identical to that described in the above paper. If legacy is set to False
    (default). The output will be "channel consistent", meaning you will be able
    to concat the input and upscale it for a simple enlargement. For example::

        $ zoom = 2
        $ inp = tf.concat([inputs for _ in range(0, zoom * zoom)], axis=-1)
        $ out = subpixel_upscale(inp, zoom)

    In this example, `out` would be identical to `inputs` , except it would
    be twice the width and height and each pixel would now occupy 4 pixels

    Learn more about this implementation at:

    https://neuralpixels.com/subpixel-upscaling/

    Args:
        inputs (any): A `Tensor`
        zoom (int): The amount to zoom
        name (str): Variable scope name
        trainable (bool): If True, the original subpixel kernel will be used as
            an initializer and trained from there.
        legacy (bool): Use the legacy kernel which is a DIP replacement for the
            original implementation
        weights: A custom kernel to use as initializer

    Returns:
        A `Tensor` with a shape of [bs, in_rows * zoom, in_cols * zoom, in_chan
        / (zoom * zoom)]

    Note:
        `subpixel_downscale` and `subpixel_upscale` are 1 for 1 inversely
        functional before training.. Meaning, running an input through one and
        then the other (with the same parameters), will result in an output
        identical to the input.
    """
    with tf.compat.v1.variable_scope(name):
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
                if legacy:
                    kernel[y, x, c, c * r * r + i] = 1
                else:
                    kernel[y, x, c, c + (i * out_channels)] = 1
                i += 1

        new_rows, new_cols = int(rows * r), int(cols * r)
        new_shape = [batch_size, new_rows, new_cols, out_channels]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1, r, r, 1]

        if weights is not None:
            kernel = weights
        if trainable:
            kernel_weights = tf.compat.v1.get_variable(
                name='kernel',
                shape=kernel.shape,
                initializer=tf.compat.v1.constant_initializer(kernel),
                dtype=inputs.dtype
            )
        else:
            kernel_weights = tf.constant(kernel, dtype=inputs.dtype)

        out = tf.nn.conv2d_transpose(inputs, kernel_weights, tf_shape, strides_shape, padding='VALID')

        return out


def subpixel_downscale(inputs, zoom=2, name='subpixel_downscale', trainable=False, legacy=False):
    """Subpixel Downscale

    Source for original implementation: https://arxiv.org/abs/1609.05158

    If the legacy flag is passed, the kernel generated will produce an output
    inversely identical to that described in the above paper. If legacy is set
    to False (default). The output will be "channel consistent", meaning you
    will be able to split the output for sub pixel `copies` For example::

        $ zoom = 2
        $ out = subpixel_upscale(inputs, zoom)
        $ splits = tf.split(out, num_or_size_splits=zoom * zoom, axis=-1)

    In this example, splits would be a list of 4 tensors, all identical to
    the input, except the would be 1/2 the size and split would contain
    completely separate pixel information.

    Learn more about this implementation at:

    https://neuralpixels.com/subpixel-upscaling/

    Args:
        inputs (any): A `Tensor`
        zoom (int): The amount to down sample
        name (str): Variable scope name
        trainable (bool): If True, the original subpixel kernel will be used as
            an initializer and trained from there.
        legacy (bool): Use the legacy kernel which is a DIP replacement for the
            inverse of the original implementation

    Returns:
        A `Tensor` with a shape of [bs, in_rows / zoom, in_cols / zoom, in_chan
        * (zoom * zoom)]

    Note:
        `subpixel_downscale` and `subpixel_upscale` are 1 for 1 inversely
        functional before training.. Meaning, running an input through one and
        then the other (with the same parameters), will result in an output
        identical to the input.
    """
    with tf.compat.v1.variable_scope(name):
        r = zoom
        batch_size, rows, cols, in_channels = inputs.get_shape().as_list()
        kernel_filter_size = r
        out_channels = int(in_channels * (r * r))

        kernel_shape = [kernel_filter_size, kernel_filter_size, in_channels, out_channels]
        kernel = np.zeros(kernel_shape, np.float32)

        # Build the kernel so that a 4 pixel cluster has each goes to a separate channel.
        for c in range(0, in_channels):
            i = 0
            for x, y in itertools.product(range(r), repeat=2):
                if legacy:
                    kernel[y, x, c, c * r * r + i] = 1
                else:
                    kernel[y, x, c, c + (i * in_channels)] = 1
                i += 1
        strides_shape = [1, r, r, 1]

        if trainable:
            kernel_weights = tf.compat.v1.get_variable(
                name='kernel',
                shape=kernel.shape,
                initializer=tf.compat.v1.constant_initializer(kernel),
                dtype=inputs.dtype
            )
        else:
            kernel_weights = tf.constant(kernel, dtype=inputs.dtype)

        out = tf.nn.conv2d(input=inputs, filters=kernel_weights, strides=strides_shape, padding='VALID')

        return out
