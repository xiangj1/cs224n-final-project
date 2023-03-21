from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


SUPPORTED_RNNS = {
    "lstm": tf.keras.layers.LSTMCell,
    "rnn": tf.keras.layers.SimpleRNNCell,
    "gru": tf.keras.layers.GRUCell,
}

_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_DECAY = 0.997
_CONV_FILTERS = 32


class DeepSpeech2(tf.keras.Model):
    def __init__(self, num_rnn_layers=5, rnn_type='lstm', is_bidirectional=True,
                 rnn_hidden_size=800, num_classes=29, use_bias=False, **kwargs):
        super(DeepSpeech2, self).__init__(**kwargs)
        self.num_rnn_layers = num_rnn_layers
        self.rnn_type = rnn_type
        self.is_bidirectional = is_bidirectional
        self.rnn_hidden_size = rnn_hidden_size
        self.num_classes = num_classes
        self.use_bias = use_bias

        self.conv1 = tf.keras.layers.Conv2D(
            filters=_CONV_FILTERS, kernel_size=(41, 11), strides=(2, 2),
            padding="same", use_bias=False, activation=tf.nn.relu6, trainable=False)
        self.bn1 = tf.keras.layers.BatchNormalization(
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, trainable=False)

        self.conv2 = tf.keras.layers.Conv2D(
            filters=_CONV_FILTERS, kernel_size=(21, 11), strides=(2, 1),
            padding="same", use_bias=False, activation=tf.nn.relu6, trainable=False)
        self.bn2 = tf.keras.layers.BatchNormalization(
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, trainable=False)

        self.conv3 = tf.keras.layers.Conv2D(
            filters=_CONV_FILTERS, kernel_size=(21, 11), strides=(2, 1),
            padding="same", use_bias=False, activation=tf.nn.relu6, trainable=False)
        self.bn3 = tf.keras.layers.BatchNormalization(
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, trainable=False)

        rnn_cell = SUPPORTED_RNNS[self.rnn_type]
        self.rnn_layers = []
        for i in range(self.num_rnn_layers):
            is_batch_norm = (i != 0)
            rnn_layer = tf.keras.layers.RNN(
                rnn_cell(self.rnn_hidden_size),
                return_sequences=True, trainable=False)
            if self.is_bidirectional:
                rnn_layer = tf.keras.layers.Bidirectional(rnn_layer)
            self.rnn_layers.append(rnn_layer)
            if is_batch_norm:
                bn_layer = tf.keras.layers.BatchNormalization(
                    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, trainable=False)
                self.rnn_layers.append(bn_layer)

        self.last_layer = tf.keras.layers.Dense(
            self.num_classes, use_bias=self.use_bias, activation="softmax", name='LastLayer')

    def call(self, inputs, training):
        inputs = tf.expand_dims(inputs, axis=-1)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = tf.reshape(x, [tf.shape(x)[0], -
