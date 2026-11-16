import math

import tensorflow as tf


class GraphConvolution(tf.keras.layers.Layer):
    # Graph convolution following https://arxiv.org/abs/1609.02907.
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bound = 1.0 / math.sqrt(out_features)
        self.weight = self.add_weight(
            name='weight', shape=(in_features, out_features),
            initializer=tf.keras.initializers.RandomUniform(-bound, bound),
        )
        self.bias = self.add_weight(
            name='bias', shape=(out_features,),
            initializer=tf.keras.initializers.RandomUniform(-bound, bound),
        ) if bias else None

    def call(self, x, adj):
        support = tf.matmul(x, self.weight)
        output = tf.matmul(adj, support)
        return output + self.bias if self.bias is not None else output
