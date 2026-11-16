import argparse
from pathlib import Path
import random
from random import choice, choices, randint
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from networkx.algorithms.community.modularity_max import \
    greedy_modularity_communities

def draw_graph(
    graph: nx.Graph,
    color_map: List[str] = None,
    positions: Dict[int, Tuple[float, float]] = None,
    axis_labels: Tuple[str, str] = None,
    with_edges=True,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 13))

    nx.draw(
        graph,
        nx.spring_layout(graph) if positions is None else positions,
        node_size=400,
        node_color=color_map,
        with_labels=True,
        ax=ax,
    )

    if axis_labels is not None:
        plt.axis('on')
        ax.tick_params(left=True, bottom=True,
                       labelleft=True, labelbottom=True)
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])


def create_graph() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, nx.Graph]:
    number_of_samples = 200
    probability_of_edge = 0.03
    seed = 13

    random.seed(seed)

    feature_range = {
        0: {
            0: (1, 3),
            1: (1, 3),
            2: (1, 10),
            3: (1, 10),
            4: (7, 10),
            5: (4, 6),
        },
        1: {
            0: (1, 5),
            1: (6, 10),
            2: (1, 3),
            3: (1, 5),
            4: (5, 10),
            5: (1, 3),
        },
        2: {
            0: (4, 7),
            1: (1, 3),
            2: (1, 10),
            3: (4, 6),
            4: (1, 3),
            5: (1, 10),
        },
        3: {
            0: (1, 2),
            1: (2, 6),
            2: (8, 10),
            3: (8, 10),
            4: (8, 10),
            5: (4, 10),
        },
        4: {
            0: (3, 8),
            1: (4, 6),
            2: (2, 5),
            3: (1, 3),
            4: (1, 4),
            5: (1, 5),
        },
        5: {
            0: (4, 7),
            1: (5, 8),
            2: (5, 8),
            3: (7, 10),
            4: (5, 10),
            5: (3, 6),
        },
        6: {
            0: (7, 10),
            1: (9, 10),
            2: (1, 4),
            3: (1, 10),
            4: (3, 10),
            5: (6, 10),
        },
        7: {
            0: (3, 8),
            1: (7, 10),
            2: (8, 10),
            3: (5, 10),
            4: (6, 10),
            5: (1, 10),
        },
    }

    graph = nx.fast_gnp_random_graph(
        number_of_samples, probability_of_edge, seed=seed)

    communities = list(greedy_modularity_communities(graph))

    features = [[] for _ in range(number_of_samples)]
    labels = [[] for _ in range(number_of_samples)]

    for community in range(len(communities)):
        community_features = feature_range[community]
        label = 0 if community in [0, 1, 2, 4] else 1

        for v in communities[community]:
            features[v] = [
                randint(community_features[0][0], community_features[0][1]),
                randint(community_features[1][0], community_features[1][1]),
                randint(community_features[2][0], community_features[2][1]),
                randint(community_features[3][0], community_features[3][1]),
                randint(community_features[4][0], community_features[4][1]),
                randint(community_features[5][0], community_features[5][1]),
            ]
            labels[v] = choices(
                [0, 1] if label == 0 else [1, 0], weights=[95, 5])[0]

    indexes = list(range(number_of_samples))
    train_indexes = []

    while len(set(train_indexes)) <= 50:
        train_indexes.append(choice(indexes))

    train_indexes = list(set(train_indexes))

    features = tf.constant(features, dtype=tf.float32) / 10

    train_labels = tf.constant([[labels[i], i]
                                 for i in train_indexes], dtype=tf.int64)
    validation_labels = tf.constant(labels, dtype=tf.int64)

    adjacency_matrix = tf.convert_to_tensor(nx.to_numpy_array(graph))

    return features, train_labels, validation_labels, adjacency_matrix, graph

class GraphConv(tf.keras.layers.Layer):
    def __init__(self, input_features, output_features):
        super().__init__(dtype=tf.float64)
        self.weight = self.add_weight(name='weight', shape=(input_features, output_features),
                                      initializer='glorot_uniform')
        self.bias = self.add_weight(name='bias', shape=(output_features,), initializer='zeros')

    def call(self, inputs, adjacency_matrix):
        return adjacency_matrix @ inputs @ self.weight + self.bias


class GCN(tf.keras.Model):
    def __init__(self):
        super().__init__(dtype=tf.float64)
        self._graph_conv_1 = GraphConv(6, 4)
        self._graph_conv_2 = GraphConv(4, 2)

    def call(self, inputs, adjacency_matrix):
        hidden = tf.nn.relu(self._graph_conv_1(inputs, adjacency_matrix))
        return tf.nn.log_softmax(self._graph_conv_2(hidden, adjacency_matrix), axis=1)


def normalize_adjacency(adjacency):
    with_self_loops = adjacency + tf.eye(len(adjacency), dtype=adjacency.dtype)
    inv_degree = tf.math.rsqrt(tf.reduce_sum(with_self_loops, axis=1))
    return inv_degree[:, None] * with_self_loops * inv_degree[None, :]


def nll_loss(log_probs, labels):
    return -tf.reduce_mean(tf.gather(log_probs, labels, axis=1, batch_dims=1))


def accuracy(log_probs, labels):
    return float(tf.reduce_mean(tf.cast(tf.argmax(log_probs, axis=1) == labels, tf.float64)))


def train(model, features, train_labels, labels, adjacency, epochs):
    optimizer = tf.keras.optimizers.Adam(epsilon=1e-8)
    train_nodes = train_labels[:, 1]
    validation = ~np.isin(np.arange(len(labels)), train_nodes.numpy())
    validation_labels = tf.boolean_mask(labels, validation)
    print(f'Train nodes: {len(train_nodes)}, validation nodes: {len(validation_labels)}', flush=True)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            output = tf.gather(model(features, adjacency), train_nodes)
            loss = nll_loss(output, train_labels[:, 0])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    records = []
    for epoch in range(1, epochs + 1):
        train_loss = float(train_step())
        output = model(features, adjacency)
        validation_output = tf.boolean_mask(output, validation)
        validation_loss = float(nll_loss(validation_output, validation_labels))
        validation_accuracy = accuracy(validation_output, validation_labels)
        train_accuracy = accuracy(tf.gather(output, train_nodes), train_labels[:, 0])
        records.append((epoch, train_loss, train_accuracy, validation_loss, validation_accuracy))
        if epoch in (1, epochs) or epoch % 100 == 0:
            print(f'Epoch {epoch}: train accuracy={train_accuracy:.2%}, '
                  f'validation loss={validation_loss:.6f}, validation accuracy={validation_accuracy:.2%}', flush=True)
    return pd.DataFrame(records, columns=['epoch', 'train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy'])


def main():
    parser = argparse.ArgumentParser(description='Train a GCN to classify nodes of a synthetic graph.')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(13)
    features, train_labels, validation_labels, adjacency_matrix, G = create_graph()
    features = tf.cast(features, tf.float64)
    adjacency = normalize_adjacency(tf.cast(adjacency_matrix, tf.float64))
    model = GCN()
    history = train(model, features, train_labels, validation_labels, adjacency, args.epochs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(args.output, index=False)
    if args.plot:
        plot_example(model, features, train_labels, validation_labels, adjacency, G, history)


def plot_example(model, features, train_labels, validation_labels, adjacency, G, history):
    import plotly.express as px

    feature_values = features.numpy()
    colors = ['gray', 'purple']
    color_map_train = ['red' for _ in range(200)]
    color_map = [colors[i] for i in validation_labels.numpy()]

    for i in train_labels.numpy():
        color_map_train[i[1]] = colors[i[0]]

    draw_graph(G)

    draw_graph(G, color_map_train)


    draw_graph(G, color_map)

    positions = {i: (feature_values[i, 1], feature_values[i, 2]) for i in range(len(features))}

    draw_graph(G, color_map, positions, axis_labels=('love', 'wealth'))

    positions = {i: (feature_values[i, 0], feature_values[i, 5]) for i in range(len(features))}

    draw_graph(
        G, color_map, positions, axis_labels=('number of dogs', 'age of customer'))

    positions = {i: (feature_values[i, 3], feature_values[i, 4]) for i in range(len(features))}

    draw_graph(G, color_map, positions, axis_labels=(
        'city size', 'healthy eating'))

    probability = tf.exp(model(features, adjacency))
    certainty = tf.reduce_max(probability, axis=1)
    predicted = tf.argmax(probability, axis=1)
    positions = {i: (float(predicted[i]), float(certainty[i])) for i in range(len(features))}
    draw_graph(G, color_map, positions, axis_labels=('junk, premium', 'certainty'))
    plt.show()
    long_history = history.melt(id_vars='epoch', var_name='metric', value_name='value')
    px.line(long_history, x='epoch', y='value', facet_row='metric').show()


if __name__ == '__main__':
    main()
