from pathlib import Path

import tensorflow as tf

from data.datasets import Traffic_inflow


DATA_DIR = Path(__file__).resolve().parent


class TrafficBatches:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        fields = (dataset.X, dataset.Y)
        if not dataset.is_train:
            fields += (dataset.Y_original,)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        self.batches = tf.data.Dataset.from_tensor_slices(fields).batch(batch_size).with_options(options)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _make_loaders(filename, time_interval, time_lag, tg_in_one_day, forecast_day_number, pre_len, batch_size):
    data = Traffic_inflow.read_data(DATA_DIR / filename)
    datasets = [
        Traffic_inflow(time_interval, time_lag, tg_in_one_day, forecast_day_number, data, pre_len,
              is_train=is_train, is_val=is_val)
        for is_train, is_val in ((True, False), (True, True), (False, False))
    ]
    maximum, minimum = datasets[0].get_max_min_inflow()
    loaders = [TrafficBatches(split, batch_size) for split in datasets]
    return *loaders, maximum, minimum


def get_inflow_dataloader(time_interval=30, time_lag=5, tg_in_one_day=36, forecast_day_number=5,
                         pre_len=1, batch_size=8):
    return _make_loaders('in_15min.csv', time_interval, time_lag, tg_in_one_day, forecast_day_number, pre_len, batch_size)


def get_outflow_dataloader(time_interval=15, time_lag=5, tg_in_one_day=72, forecast_day_number=5,
                         pre_len=1, batch_size=8):
    return _make_loaders('out_15min.csv', time_interval, time_lag, tg_in_one_day, forecast_day_number, pre_len, batch_size)
