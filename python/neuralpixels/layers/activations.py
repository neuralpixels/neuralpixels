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
