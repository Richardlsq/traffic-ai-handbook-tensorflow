import argparse
import csv
from datetime import datetime
from hashlib import file_digest
import json
from math import isfinite
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.record_writer import RecordWriter

from data.get_dataloader import get_speed_dataloader
from model.main_model import Model
from utils.earlystopping import EarlyStopping, load_checkpoint_metadata
from utils.metrics import Metrics
from utils.utils import GetLaplacian


PROJECT = Path(__file__).resolve().parents[1]
NODES = 156
ADJACENCY = PROJECT / 'data' / 'sz_adj1.csv'
DATA_FILES = [ADJACENCY, PROJECT / 'data' / 'sz_speed-论文数据.csv']
DEFAULT_EPOCHS = 5000


def _device(name):
    kind, _, index_text = name.lower().lstrip('/').partition(':')
    if kind == 'auto':
        kind = 'gpu' if tf.config.list_physical_devices('GPU') else 'cpu'
    if kind == 'cuda':
        kind = 'gpu'
    if kind not in ('cpu', 'gpu') or (index_text and not index_text.isdecimal()):
        raise ValueError('Use --device auto, cpu, gpu, or cuda:<index>')
    index = int(index_text or 0)
    if index >= len(tf.config.list_physical_devices(kind.upper())):
        raise ValueError(f'TensorFlow device {kind}:{index} is unavailable')
    if kind == 'cpu':
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    return f'/{kind.upper()}:{index}'


def _config(holdout_days, batch_size):
    return {
        'time_interval': 15, 'time_lag': 10, 'tg_in_one_day': 72,
        'forecast_day_number': holdout_days, 'pre_len': 1, 'batch_size': batch_size,
    }


def _setup(config, device):
    *loaders, maximum, minimum = get_speed_dataloader(**config)
    splits = [(loader,) for loader in loaders]
    adjacency = GetLaplacian(np.loadtxt(ADJACENCY, delimiter=',')).get_normalized_adj(NODES)
    with tf.device(device):
        graph = tf.convert_to_tensor(adjacency)
        model = Model(config['time_lag'], config['pre_len'], NODES)
        sample = tf.zeros((1, NODES, config['time_lag']))
        model(sample, graph)
    return splits, (maximum, minimum), graph, model


def _data_hashes():
    fingerprints = {}
    for path in DATA_FILES:
        with path.open('rb') as stream:
            fingerprints[path.name] = file_digest(stream, 'sha256').hexdigest()
    return fingerprints


def _batches(loaders):
    for batches in zip(*loaders, strict=True):
        yield (batches[0][0],), batches[0][1], batches[0][2:]


def _make_train_step(model, graph, optimizer):
    @tf.function(reduce_retracing=True)
    def train_step(features, target):
        with tf.device(graph.device), tf.GradientTape() as tape:
            prediction = model(*features, graph, training=True)
            error = tf.reduce_sum(tf.math.squared_difference(prediction, target))
            loss = error / tf.cast(tf.size(target), error.dtype)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return error
    return train_step


def _train_epoch(train_step, loaders):
    sum_squared = 0.0
    count = 0
    for features, target, _ in _batches(loaders):
        sum_squared += float(train_step(features, target).numpy())
        count += int(tf.size(target))
    return sum_squared / count


def _validation_mse(model, validation_loaders, graph):
    sum_squared = 0.0
    count = 0
    with tf.device(graph.device):
        for features, target, _ in _batches(validation_loaders):
            prediction = model(*features, graph, training=False)
            error = tf.math.squared_difference(tf.cast(prediction, tf.float64), tf.cast(target, tf.float64))
            sum_squared += float(tf.reduce_sum(error).numpy())
            count += int(tf.size(target))
    return sum_squared / count


def _metrics(actual, forecast):
    def by_node(values):
        return values.transpose(0, 2, 1).reshape(-1, values.shape[1])

    rmse, r2, mae, wmape = Metrics(by_node(actual), by_node(forecast)).evaluate_performance()
    return {'rmse': rmse, 'mae': mae, 'r2': r2 if isfinite(r2) else None,
            'wmape': wmape if isfinite(wmape) else None}


def _evaluate(model, loaders, graph, scaling, output, checkpoint_epoch):
    maximum, minimum = scaling
    span = maximum - minimum or 1.0
    predictions, observations, reference = [], [], []
    sum_squared = 0.0
    count = 0
    with tf.device(graph.device):
        for features, target, (original,) in _batches(loaders):
            prediction = model(*features, graph, training=False).numpy().astype(np.float64)
            normalized = target.numpy()
            sum_squared += np.square(prediction - normalized).sum()
            count += normalized.size
            predictions.append(prediction * span + minimum)
            observations.append(original.numpy())
            last = features[0][:, :, -1:].numpy().astype(np.float64) * span + minimum
            reference.append(np.broadcast_to(last, original.shape))
    predicted = np.maximum(np.concatenate(predictions), 0.0)
    actual = np.concatenate(observations)
    persistence = np.concatenate(reference)
    indices = loaders[0].dataset.target_indices
    result = {
        'checkpoint_epoch': checkpoint_epoch, 'split_metadata_verified': checkpoint_epoch is not None,
        'samples': len(indices), 'first_target_index': int(indices[0]), 'last_target_index': int(indices[-1]),
        'normalized_mse': float(sum_squared / count), 'model': _metrics(actual, predicted),
        'last_observation': _metrics(actual, persistence),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'predictions.npz', target_index=indices, actual=actual,
                        predicted=predicted, last_observation=persistence)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


def train_main():
    parser = argparse.ArgumentParser(description='Train on chronological data and save the best validation checkpoint.')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--holdout-days', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--tensorboard', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.holdout_days) < 1 or not isfinite(args.lr) or args.lr <= 0:
        parser.error('epochs, patience, lr, batch size and holdout days must be positive')

    device = _device(args.device)
    tf.keras.utils.set_random_seed(args.seed)
    config = _config(args.holdout_days, args.batch_size)
    (train, validation, test), scaling, graph, model = _setup(config, device)
    with tf.device(device):
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, epsilon=1e-8)
    train_step = _make_train_step(model, graph, optimizer)
    output = args.output_dir or PROJECT / 'outputs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    stopper = EarlyStopping(patience=args.patience, checkpoint_name='best.weights.h5')
    fingerprints = _data_hashes()
    writer = None
    if args.tensorboard:
        log_dir = output / 'tensorboard'
        log_dir.mkdir()
        timestamp = datetime.now().timestamp()
        log_path = log_dir / f'events.out.tfevents.{int(timestamp)}.course'
        # Python file I/O also supports non-ASCII paths on native Windows.
        writer = RecordWriter(log_path.open('wb'))
        writer.write(Event(wall_time=timestamp, file_version='brain.Event:2').SerializeToString())
        writer.flush()

    splits = {}
    for name, loaders in zip(('train', 'validation', 'test'), (train, validation, test), strict=True):
        indices = loaders[0].dataset.target_indices
        splits[name] = {'samples': len(indices), 'start': int(indices[0]), 'end': int(indices[-1])}
    print(f'Device: {device}. Splits: {json.dumps(splits)}', flush=True)
    try:
        with (output / 'training.csv').open('w', newline='', encoding='utf-8') as stream:
            history = csv.writer(stream)
            history.writerow(('epoch', 'train_mse', 'validation_mse'))
            for epoch in range(1, args.epochs + 1):
                train_mse = _train_epoch(train_step, train)
                val_mse = _validation_mse(model, validation, graph)
                payload = {
                    'config': config, 'splits': splits, 'data_sha256': fingerprints,
                    'graph_project': 'gcn', 'seed': args.seed, 'epoch': epoch, 'validation_mse': val_mse,
                    'maximum': scaling[0], 'minimum': scaling[1],
                }
                stopper(val_mse, payload, model, epoch, output)
                history.writerow((epoch, train_mse, val_mse))
                stream.flush()
                if writer:
                    summary = Summary(value=[
                        Summary.Value(tag='loss_train', simple_value=train_mse),
                        Summary.Value(tag='loss_validation', simple_value=val_mse),
                    ])
                    event = Event(wall_time=datetime.now().timestamp(), step=epoch, summary=summary)
                    writer.write(event.SerializeToString())
                    writer.flush()
                print(f'Epoch {epoch}: train MSE={train_mse:.8f}, validation MSE={val_mse:.8f}', flush=True)
                if stopper.early_stop:
                    break
    finally:
        if writer:
            writer.close()
    print(f'Best epoch: {stopper.best_epoch}; validation MSE: {stopper.val_loss_min:.8f}', flush=True)
    print(f'Checkpoint: {output / "best.weights.h5"}', flush=True)


def predict_main():
    parser = argparse.ArgumentParser(description='Evaluate a checkpoint on its chronological test split.')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = _device(args.device)
    checkpoint = load_checkpoint_metadata(args.checkpoint)
    if checkpoint.get('graph_project') != 'gcn':
        parser.error('Choose a checkpoint produced by this project')
    if checkpoint['data_sha256'] != _data_hashes():
        raise ValueError('Data or graph files differ from the saved training run')
    (_, validation, test), scaling, graph, model = _setup(checkpoint['config'], device)
    if scaling != (checkpoint['maximum'], checkpoint['minimum']):
        raise ValueError('Data scaling differs from the saved training run')
    test_indices = test[0].dataset.target_indices
    saved_test = checkpoint['splits']['test']
    if (len(test_indices), test_indices[0], test_indices[-1]) != (saved_test['samples'], saved_test['start'], saved_test['end']):
        raise ValueError('The test split differs from the saved training run')
    model.load_weights(args.checkpoint)
    val_mse = _validation_mse(model, validation, graph)
    print(f'Checkpoint epoch {checkpoint["epoch"]}: saved validation MSE={checkpoint["validation_mse"]:.8f}, '
          f'recomputed validation MSE={val_mse:.8f}', flush=True)
    _evaluate(model, test, graph, scaling, args.output_dir or args.checkpoint.parent / 'test', checkpoint['epoch'])
