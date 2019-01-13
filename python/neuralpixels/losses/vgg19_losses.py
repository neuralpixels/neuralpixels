import tensorflow as tf
from collections import OrderedDict
from neuralpixels.pretrained import vgg19

STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_1')
CONTENT_LAYERS = ('conv4_2', 'conv5_2')


def tensor_size(tensor):
    height = tf.shape(tensor)[-3]
    width = tf.shape(tensor)[-2]
    filters = tf.shape(tensor)[-1]
    return height * width * filters


def _convert_to_gram_matrix(inputs):
    shape = tf.shape(inputs)
    batch, height, width, filters = shape[0], shape[1], shape[2], shape[3]
    _size = height * width * filters
    size = tf.cast(_size, inputs.dtype)

    feats = tf.reshape(inputs, (batch, height * width, filters))
    feats_t = tf.transpose(feats, perm=[0, 2, 1])
    grams_raw = tf.matmul(feats_t, feats)
    gram_matrix = tf.divide(grams_raw, size)
    return gram_matrix


def style_loss(targets, predictions, style_layers=STYLE_LAYERS):
    if isinstance(targets, OrderedDict):
        targets_net = targets
    else:
        targets_net = vgg19.process(targets)

    if isinstance(predictions, OrderedDict):
        predictions_net = predictions
    else:
        predictions_net = vgg19.process(predictions)

    dtype = targets_net['input'].dtype
    total_style_loss = tf.constant(0.0, dtype=dtype)

    for style_layer in style_layers:
        pred_grams = _convert_to_gram_matrix(predictions_net[style_layer])
        target_grams = _convert_to_gram_matrix(targets_net[style_layer])

        content_size = tf.cast(tensor_size(targets_net[style_layer]), dtype=dtype)

        # don't sum the batch, keep separate images separate. Not used here, but done if this is used elsewhere
        def seperated_loss(y_pred, y_true):
            sum_axis = [1, 2]
            diff = tf.abs(y_pred - y_true)
            l2 = tf.reduce_sum(diff ** 2, axis=sum_axis) / 2
            return 2. * l2 / content_size

        pred_itemized_loss = seperated_loss(pred_grams, target_grams)
        layer_loss = tf.reduce_mean(pred_itemized_loss * vgg19.layer_weights[style_layer]['style'])

        # add this layer loss to the total loss
        total_style_loss += layer_loss

    # return avg layer loss
    return total_style_loss / float(len(style_layers))


def content_loss(targets, predictions, content_layers=CONTENT_LAYERS):
    if isinstance(targets, OrderedDict):
        targets_net = targets
    else:
        targets_net = vgg19.process(targets)

    if isinstance(predictions, OrderedDict):
        predictions_net = predictions
    else:
        predictions_net = vgg19.process(predictions)

    dtype = targets_net['input'].dtype
    total_content_loss = tf.constant(0.0, dtype=dtype)

    for content_layer in content_layers:
        pred_layer = predictions_net[content_layer]
        target_layer = targets_net[content_layer]

        content_size = tf.cast(tensor_size(targets_net[content_layer]), dtype=dtype)

        # don't sum the batch, keep separate images separate. Not used here, but done if this is used elsewhere
        def seperated_loss(y_pred, y_true):
            sum_axis = [1, 2, 3]
            diff = tf.abs(y_pred - y_true)
            l2 = tf.reduce_sum(diff ** 2, axis=sum_axis) / 2
            return 2. * l2 / content_size

        pred_itemized_loss = seperated_loss(pred_layer, target_layer)
        layer_loss = tf.reduce_mean(pred_itemized_loss * vgg19.layer_weights[content_layer]['content'])

        # add this layer loss to the total loss
        total_content_loss += layer_loss

    # return avg layer loss
    return total_content_loss / float(len(content_layers))
