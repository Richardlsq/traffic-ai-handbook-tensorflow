from pathlib import Path

import numpy as np
import tensorflow as tf


class Traffic_speed:
    def __init__(self, time_interval, time_lag, tg_in_one_day, forecast_day_number,
                 speed_data, pre_len, is_train=True, is_val=False, val_rate=0.1):
        self.is_train = is_train
        data = self.read_data(speed_data)
        total_steps = data.shape[1]
        first_target = time_lag
        if min(time_lag, pre_len, tg_in_one_day, forecast_day_number) < 1 or not 0 <= val_rate < 1:
            raise ValueError('Invalid window, horizon, or split configuration')
        test_start = total_steps - tg_in_one_day * forecast_day_number
        pretest_stop = test_start - pre_len + 1
        pretest_windows = pretest_stop - first_target
        if pretest_windows < 1 or total_steps - test_start < pre_len:
            raise ValueError('Series is too short for the requested splits and windows')
        val_count = max(1, int(pretest_windows * val_rate)) if val_rate else 0
        val_start = pretest_stop - val_count if val_count else test_start
        # Target intervals remain disjoint for horizons greater than one.
        if val_start - pre_len < first_target:
            raise ValueError('Training windows would be empty')
        train_data = data[:, :val_start]
        self.minimum = float(train_data.min())
        self.maximum = float(train_data.max())
        scale = self.maximum - self.minimum or 1.0
        normalized = ((data - self.minimum) / scale).astype(np.float32)

        if not is_train:
            start, stop = test_start, total_steps - pre_len + 1
        elif is_val:
            start, stop = val_start, pretest_stop
        else:
            start, stop = first_target, val_start - pre_len + 1
        if start >= stop:
            raise ValueError('The selected split has no valid prediction windows')
        self.target_indices = np.arange(start, stop)
        offsets = np.arange(-time_lag, 0)
        indices = self.target_indices[:, None] + offsets
        histories = normalized[:, indices].transpose(1, 0, 2)
        targets = self.target_indices[:, None] + np.arange(pre_len)
        self.X = tf.convert_to_tensor(np.ascontiguousarray(histories))
        self.Y = tf.convert_to_tensor(np.ascontiguousarray(normalized[:, targets].transpose(1, 0, 2)))
        if not is_train:
            self.Y_original = tf.convert_to_tensor(np.ascontiguousarray(data[:, targets].transpose(1, 0, 2)))

    @staticmethod
    def read_data(source):
        if isinstance(source, (str, Path)):
            return np.loadtxt(source, delimiter=',', ndmin=2).T
        return source

    def get_max_min_speed(self):
        return self.maximum, self.minimum

    def __getitem__(self, item):
        if self.is_train:
            return self.X[item], self.Y[item]
        return self.X[item], self.Y[item], self.Y_original[item]

    def __len__(self):
        return len(self.X)
