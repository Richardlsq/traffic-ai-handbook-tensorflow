import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py


METADATA_ATTRIBUTE = 'course_metadata'


def load_checkpoint_metadata(path):
    if not path.name.endswith('.weights.h5'):
        raise ValueError('Choose a .weights.h5 checkpoint trained in this edition.')
    with h5py.File(path, 'r') as checkpoint:
        if METADATA_ATTRIBUTE not in checkpoint.attrs:
            raise ValueError('This checkpoint has no course training metadata.')
        return json.loads(checkpoint.attrs[METADATA_ATTRIBUTE])


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, checkpoint_name=None):
        self.patience = patience
        self.verbose = verbose
        self.checkpoint_name = checkpoint_name
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_epoch = None

    def __call__(self, val_loss, metadata, model, epoch, save_path):
        if not isfinite(val_loss):
            raise ValueError(f'Non-finite validation loss at epoch {epoch}: {val_loss}')
        if val_loss < self.val_loss_min:
            self.save_checkpoint(val_loss, metadata, model, epoch, save_path)
            self.counter = 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience

    def save_checkpoint(self, val_loss, metadata, model, epoch, save_path):
        filename = self.checkpoint_name or f'model_checkpoint_{epoch}_{val_loss:.8f}.weights.h5'
        location = Path(save_path) / filename
        with TemporaryDirectory(dir=location.parent, prefix='.checkpoint-') as directory:
            temporary = Path(directory) / filename
            model.save_weights(temporary)
            with h5py.File(temporary, 'a') as checkpoint:
                checkpoint.attrs[METADATA_ATTRIBUTE] = json.dumps(metadata, allow_nan=False)
            temporary.replace(location)
        if self.verbose:
            print(f'Validation MSE improved from {self.val_loss_min:.8f} to {val_loss:.8f}', flush=True)
        self.val_loss_min = val_loss
        self.best_epoch = epoch
