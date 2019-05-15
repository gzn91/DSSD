import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np

def fc(x, units, name, activation_fn=tf.nn.relu):
    with tf.variable_scope(name):
        nin = x.get_shape().as_list()[-1]
        scale = np.sqrt(2. / (nin + units))
        w = tf.get_variable(shape=(nin, units), initializer=tc.layers.xavier_initializer(), name='w')
        b = tf.get_variable(shape=(1, units), initializer=tf.zeros_initializer(), name='b')

        h = tf.matmul(x, w) + b
        h_activ = activation_fn(h)

        return h_activ


def conv2d(x, filters, kernel_size=3, name='', strides=1, padding='VALID',
           activation_fn=tf.nn.leaky_relu, training=False, use_bn=False, dilations=1, l2reg=0.0):
    nb, nw, nh, nc = x.get_shape().as_list()
    scale = np.sqrt(2. / (filters + nc))
    with tf.variable_scope(name):
        b = 0.
        wx = tf.get_variable("w", [kernel_size, kernel_size, nc, filters], initializer=tc.layers.xavier_initializer_conv2d())
        if not use_bn:
            b = tf.get_variable("b", [1, 1, 1, filters], initializer=tf.constant_initializer(0.))
        x = tf.nn.conv2d(x, filter=wx, strides=[1, strides, strides, 1], padding=padding,
                         dilations=[1, dilations, dilations,1]) + b

        tf.add_to_collections(tf.GraphKeys.REGULARIZATION_LOSSES, l2reg*tf.nn.l2_loss(wx))

        if use_bn:
            x = tf.layers.batch_normalization(x, training=training)

    return activation_fn(x)


def conv2d_transpose(x, filters, kernel_size=3, name='', strides=2, padding='VALID',
                     activation_fn=tf.nn.leaky_relu, training=False, use_bn=False, l2reg=0.0):
    nb, nw, nh, nc = x.get_shape().as_list()
    scale = np.sqrt(2. / (filters + nc))
    if padding == 'VALID':
        out_h = nh * strides + max(kernel_size - strides, 0)
        out_w = nw * strides + max(kernel_size - strides, 0)
    elif padding == 'SAME':
        out_h = nh * strides
        out_w = nw * strides
    else:
        raise NameError("padding must be 'SAME' or 'VALID'")
    out_shape = [tf.shape(x)[0], out_h, out_w, filters]

    with tf.variable_scope(name):
        b = 0.
        wx = tf.get_variable("w", [kernel_size, kernel_size, filters, nc], initializer=tc.layers.xavier_initializer_conv2d())
        if not use_bn:
            b = tf.get_variable("b", [1, 1, 1, filters], initializer=tf.constant_initializer(0.))
        x = tf.nn.conv2d_transpose(x, filter=wx, output_shape=out_shape,
                                   strides=[1, strides, strides, 1], padding=padding) + b
        tf.add_to_collections(tf.GraphKeys.REGULARIZATION_LOSSES, l2reg*tf.nn.l2_loss(wx))
        if use_bn:
            x = tf.layers.batch_normalization(x, training=training)
    return activation_fn(x)


def max_pool2d(x, name, kernel_size=2, strides=2, pad='SAME'):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, strides, strides, 1], padding=pad)


def l2_normalization(x, scale=20.):
    shape = x.get_shape().as_list()[-1]
    gamma = scale*tf.get_variable(name='scale', shape=(shape,), initializer=tf.ones_initializer, dtype=tf.float32)

    x_norm = tf.nn.l2_normalize(x, axis=-1)

    return gamma * x_norm
