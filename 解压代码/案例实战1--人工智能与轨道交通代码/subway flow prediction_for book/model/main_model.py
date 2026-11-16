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
        self.GCN_week = GraphConvolution(time_lag, time_lag)
        self.GCN_day = GraphConvolution(time_lag, time_lag)
        self.GCN_time = GraphConvolution(time_lag, time_lag)
        self.Conv2D = tf.keras.layers.Conv2D(8, 3, padding='same', kernel_initializer='glorot_normal')
        self.flatten = tf.keras.layers.Flatten()
        self.linear1 = _dense(8 * 3 * time_lag * station_num, 1024, 'relu')
        self.linear2 = _dense(1024, 512, 'relu')
        self.linear3 = _dense(512, station_num * pre_len)

    def _encode_history(self, flow, adj):
        week, day, recent = tf.split(flow, 3, axis=-1)
        return tf.concat((
            self.GCN_week(week, adj),
            self.GCN_day(day, adj),
            self.GCN_time(recent, adj),
        ), axis=-1)

    def call(self, inflow, outflow, adj, training=False):
        inflow_features = self._encode_history(inflow, adj)
        # Mask the second encoded channel while keeping the same model and initialization.
        outflow_features = (
            tf.zeros_like(inflow_features) if outflow is None else self._encode_history(outflow, adj)
        )
        inputs = tf.stack((inflow_features, outflow_features), axis=-1)
        # Preserve the dense layer's channels-first feature ordering.
        features = self.flatten(tf.transpose(self.Conv2D(inputs), (0, 3, 1, 2)))
        features = self.linear2(self.linear1(features))
        return tf.reshape(self.linear3(features), (-1, self.station_num, self.pre_len))
