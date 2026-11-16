import tensorflow as tf


class GraphAttentionLayer(tf.keras.layers.Layer):
    # Dense graph attention following https://arxiv.org/abs/1710.10903.
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.W = self._weight('W', (in_features, out_features))
        self.a = self._weight('a', (2 * out_features, 1))

    def _weight(self, name, shape):
        initializer = tf.keras.initializers.VarianceScaling(
            scale=1.414 ** 2, mode='fan_avg', distribution='uniform',
        )
        return self.add_weight(name=name, shape=shape, initializer=initializer)

    def call(self, x, adj, training=False):
        h = tf.matmul(x, self.W)
        # aᵀ[hᵢ || hⱼ] = a_leftᵀhᵢ + a_rightᵀhⱼ.
        left, right = tf.split(self.a, 2, axis=0)
        scores = tf.matmul(h, left) + tf.linalg.matrix_transpose(tf.matmul(h, right))
        scores = tf.nn.leaky_relu(scores, alpha=self.alpha)
        scores = tf.where(adj > 0, scores, tf.cast(-9e15, scores.dtype))
        attention = self.dropout(tf.nn.softmax(scores, axis=-1), training=training)
        output = tf.matmul(attention, h)
        return tf.nn.elu(output) if self.concat else output


class GAT(tf.keras.Model):
    def __init__(self, nfeat, nhid, n_final_out, dropout, alpha, nheads):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, dropout, alpha) for _ in range(nheads)
        ]
        self.out_att = GraphAttentionLayer(nhid * nheads, n_final_out, dropout, alpha, concat=False)

    def call(self, x, adj, training=False):
        x = self.dropout(x, training=training)
        x = tf.concat([head(x, adj, training=training) for head in self.attentions], axis=-1)
        x = self.dropout(x, training=training)
        return tf.nn.elu(self.out_att(x, adj, training=training))
