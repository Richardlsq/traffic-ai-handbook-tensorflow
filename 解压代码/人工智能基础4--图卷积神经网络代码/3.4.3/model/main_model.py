from math import sqrt

import tensorflow as tf

from model.GCN_layers import GraphConvolution


def _dense(in_features, out_features, activation=None):
    bound = 1.0 / sqrt(in_features)
    return tf.keras.layers.Dense(
        out_features, activation=activation,
        kernel_initializer=tf.keras.initializers.RandomUniform(-bound, bound),
        bias_initializer=tf.keras.initializers.RandomUniform(-bound, bound),
    )


class Model(tf.keras.Model):
    def __init__(self, time_lag, pre_len, station_num):
        super().__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.GCN1 = GraphConvolution(time_lag, time_lag)
        self.GCN2 = GraphConvolution(time_lag, time_lag)
        self.flatten = tf.keras.layers.Flatten()
        self.linear1 = _dense(time_lag * station_num, 1024, 'relu')
        self.linear2 = _dense(1024, 512, 'relu')
        self.linear3 = _dense(512, station_num * pre_len)

    def call(self, speed, adj, training=False):
        speed = self.GCN1(speed, adj)
        features = self.flatten(self.GCN2(speed, adj))
        features = self.linear2(self.linear1(features))
        return tf.reshape(self.linear3(features), (-1, self.station_num, self.pre_len))
