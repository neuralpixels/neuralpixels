import tensorflow as tf


def residual_blending(inputs1, inputs2, name='res_blend', shift=0.0, scale=0.5):
    """
    Residual Blending Layer

    Residual blending adds two inputs together after shifting and scaling each layer.
    The shift and scale values are trainable and are seperate values for each channel.

    :param inputs1: A `Tensor`.
    :param inputs2: A `Tensor`.
    :param name: Variable scope name for weights
    :param shift: Initial shift for both inputs
    :param scale: Initial scale for both inputs
    :return: A `Tensor` with the same shape as the inputs.
    """
    with tf.variable_scope(name):
        channels = inputs1.get_shape()[-1]
        shift_var1 = tf.get_variable(
            name='shift1',
            shape=channels,
            initializer=tf.constant_initializer(shift)
        )
        scale_var1 = tf.get_variable(
            name="scale1",
            shape=channels,
            initializer=tf.constant_initializer(scale)
        )
        shift_var2 = tf.get_variable(
            name='shift2',
            shape=channels,
            initializer=tf.constant_initializer(shift)
        )
        scale_var2 = tf.get_variable(
            name="scale2",
            shape=channels,
            initializer=tf.constant_initializer(scale)
        )
        outputs_1 = (inputs1 + shift_var1) * scale_var1
        outputs_2 = (inputs2 + shift_var2) * scale_var2

        return tf.add(outputs_1, outputs_2)
