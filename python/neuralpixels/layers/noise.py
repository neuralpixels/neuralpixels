import tensorflow as tf


def noise_embed(inputs, mean=0.0, stddev=0.1, name='noise', trainable=False):
    """
    Creates a noise embedding layer. If `trainable` is set `True`,
    then it will create a trainable noise embedding that trains the mean and stddev on a
    per channel basis.

    :param inputs: Tensor to add noise to
    :type inputs: Any
    :param mean: initial mean
    :type mean: float
    :param stddev: initial stddev
    :type stddev: float
    :param name: scope name
    :type name: str
    :param trainable: If true, the mean and std deviation will be trainable weights on a per channel basis
    :type trainable: bool
    :return: A tensor with the same shape as input with trained noise added to it
    """
    with tf.variable_scope(name):
        noise_shape = inputs.get_shape().as_list()
        channels = noise_shape[-1]

        if trainable:
            _mean = tf.get_variable(
                name='mean',
                shape=channels,
                dtype=inputs.dtype,
                initializer=tf.constant_initializer(mean)
            )
            _mean = tf.reshape(_mean, [1, 1, 1, channels])

            _stddev = tf.get_variable(
                name="stddev",
                shape=channels,
                dtype=inputs.dtype,
                initializer=tf.constant_initializer(stddev)
            )
            _stddev = tf.reshape(_stddev, [1, 1, 1, channels])
        else:
            _mean = mean
            _stddev = stddev

        max_noise = tf.reduce_max(tf.abs(inputs), axis=[1, 2], keepdims=True)

        raw_noise = tf.random_normal(
            shape=noise_shape,
            mean=0,
            stddev=10.0,
            dtype=inputs.dtype
        )
        scaled_noise = raw_noise * (max_noise / 10.0)

        # scale and adjust noise to weights
        noise = (scaled_noise + _mean) * _stddev
        return inputs + noise
