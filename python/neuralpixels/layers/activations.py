import tensorflow as tf


def prelu(inputs, name='prelu', alpha=0.25):
    """Parametric Rectified Linear Unit.

    :param inputs: A `Tensor`.
    :param name: A name for the operation and variable scope.
    :param alpha: Alpha initializer for weights

    :return: A `Tensor`. Has the same type as `features`.
    """
    with tf.variable_scope(name):
        shape = inputs.get_shape()[-1]
        alphas = tf.get_variable(
            name="alpha",
            shape=shape,
            initializer=tf.constant_initializer(alpha),
            dtype=tf.float32
        )
        pos = tf.nn.relu(inputs)
        neg = alphas * (inputs - abs(inputs)) * 0.5
        return pos + neg


def prelu_clipped(inputs, name='prelu_clip', alpha=0.25, clip=6.0, epsilon=1e-5):
    """Parametric Rectified Linear Unit with channel specific trained clipping

    :param inputs: A `Tensor`.
    :param name: A name for the operation and variable scope.
    :param alpha: Alpha initializer for weights
    :param clip: Initial clip value

    :return: A `Tensor`. Has the same type as `features`.
    """
    with tf.variable_scope(name):
        base_clip = 10.00001
        shape = inputs.get_shape()[-1]
        alphas = tf.get_variable(
            name="alpha",
            shape=shape,
            initializer=tf.constant_initializer(alpha),
            dtype=tf.float32
        )
        clips = tf.get_variable(
            name="clip",
            shape=shape,
            initializer=tf.constant_initializer(clip),
            dtype=tf.float32
        )
        eps_clips = clips + epsilon
        mul = base_clip / eps_clips
        pre_clip_inputs = inputs * mul
        unscaled_clipped_inputs = tf.clip_by_value(pre_clip_inputs, -base_clip, base_clip)
        clipped_inputs = unscaled_clipped_inputs / mul
        pos = tf.nn.relu(clipped_inputs)
        neg = alphas * (clipped_inputs - abs(clipped_inputs)) * 0.5

        return pos + neg
