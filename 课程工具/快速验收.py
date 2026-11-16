"""Run short, CPU-compatible checks from the course workspace."""
import argparse
from datetime import datetime
import importlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / '解压代码'
COURSE_OUTPUT = ROOT / '课程输出'
BASIC = EXTRACTED / 'Python基础知识代码' / 'Python代码'
UNITS = ('python', 'tensorflow', 'bp', 'rnn', 'cnn', 'gcn', 'metro')


def check_environment():
    print(f'Python {sys.version.split()[0]} | {sys.executable}', flush=True)
    for name in (
        'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'openpyxl', 'PIL',
        'jupyterlab', 'ipykernel', 'nbclient', 'nbformat', 'nbconvert', 'networkx', 'plotly', 'tensorflow', 'keras', 'h5py', 'tensorboard',
    ):
        module = importlib.import_module(name)
        print(f'{name} {getattr(module, "__version__", "")}', flush=True)
    import tensorflow as tf
    with tf.device('/CPU:0'):
        trial = tf.ones((1, 3, 3, 1))
        assert tf.keras.layers.Conv2D(1, 3)(trial).shape == (1, 1, 1, 1)
    print(f'Visible TensorFlow GPUs: {len(tf.config.list_physical_devices("GPU"))}', flush=True)
    print('PASS env: imports and CPU convolution', flush=True)


def check_python():
    import numpy as np
    import pandas as pd
    from scipy import linalg
    from sklearn.metrics import mean_absolute_error

    rows = BASIC / 'Python1.3-1.4'
    data = np.loadtxt(rows / 'test.csv', delimiter=',', dtype=int)
    frame = pd.read_csv(rows / 'test.csv', header=None)
    workbook = pd.read_excel(rows / 'table.xlsx', sheet_name='Sheet1')
    assert data.shape == (5, 3) and not workbook.empty
    np.testing.assert_array_equal(data, frame.to_numpy())
    matrix = np.array([[9, 2, 8], [2, 5, 6], [5, 1, 3]])
    np.testing.assert_allclose(matrix @ linalg.inv(matrix), np.eye(3), atol=1e-12)
    assert mean_absolute_error([1, 2, 3], [2, 2, 2]) == 2 / 3
    print('PASS python: CSV (5, 3), Sheet1, matrix inverse and MAE', flush=True)


def run_notebook(filename, out, source_cells=None, overrides=None, assertion='', setup=''):
    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    source = next(EXTRACTED.rglob(filename))
    book = nbformat.read(source, as_version=4)
    code_cells = {
        cell.metadata.get('course_cell_number', position): cell
        for position, cell in enumerate(book.cells, 1)
        if cell.cell_type == 'code'
    }
    if source_cells is None:
        source_cells = list(code_cells)
    notebook = nbformat.v4.new_notebook()
    if setup:
        notebook.cells.append(nbformat.v4.new_code_cell(setup))
    overrides = overrides or {}
    for index in source_cells:
        if index not in code_cells:
            raise ValueError(f'{filename}: missing source cell {index}')
        code = code_cells[index].source
        if index in overrides:
            old, new = overrides[index]
            if code.count(old) != 1:
                raise ValueError(f'{filename}: changed source cell {index}')
            code = code.replace(old, new, 1)
        notebook.cells.append(nbformat.v4.new_code_cell(code, metadata={'source_cell': index}))
    notebook.cells.append(nbformat.v4.new_code_cell(assertion))
    destination = out / f'{source.stem}-快速验收.ipynb'
    with TemporaryDirectory(prefix='course-quick-kernel-') as directory:
        kernels = Path(directory)
        spec = kernels / 'course-quick'
        spec.mkdir()
        kernel = {
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Course quick check',
            'language': 'python',
            'env': {'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
                    'TF_NUM_INTRAOP_THREADS': '4', 'TF_NUM_INTEROP_THREADS': '2'},
        }
        (spec / 'kernel.json').write_text(json.dumps(kernel), encoding='utf-8')
        manager = KernelManager(kernel_name='course-quick',
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        try:
            NotebookClient(notebook, km=manager, timeout=120,
                           resources={'metadata': {'path': str(source.parent)}}).execute(cleanup_kc=True)
        finally:
            nbformat.write(notebook, destination)
    for cell in notebook.cells[-2:]:
        for output in cell.get('outputs', []):
            if output.output_type == 'stream' and output.name == 'stdout':
                print(output.text.rstrip(), flush=True)


def check_tensorflow(out):
    run_notebook(
        'TensorFlow2.1张量介绍.ipynb', out,
        assertion='''
tf.debugging.assert_equal(temp1, temp2)
assert tuple(output.shape) == (10, 3, 5)
assert 'CPU:0' in x.device
''',
    )
    run_notebook(
        'TensorFlow2.2数据模块.ipynb', out,
        assertion='''
assert (len(data_train), len(data_test)) == (7997, 1997)
assert tuple(xb.shape) == (5, 3) and tuple(yb.shape) == (5, 1)
tf.debugging.assert_equal(data_train[-1][1], data_train.x[-1:])
tf.debugging.assert_equal(data_test[-1][1], data_test.x[-1:])
_, last_batch = next(iter(data_loader_test.skip(len(data_loader_test) - 1)))
tf.debugging.assert_equal(last_batch[-1], data_test.x[-1:])
try:
    data_test[len(data_test)]
except IndexError:
    pass
else:
    raise AssertionError('Dataset must reject an out-of-range sample index')
''',
    )
    run_notebook(
        'TensorFlow2.3模型介绍.ipynb', out,
        assertion='''
assert tuple(out.shape) == (3, 5, 20)
assert tuple(h.shape) == tuple(c.shape) == (4, 3, 20)
assert tuple(dconv1(tf.zeros((16, 8, 1))).shape) == (16, 23, 1)
assert tuple(dconv2(tf.zeros((16, 8, 8, 1))).shape) == (16, 23, 23, 1)
assert tuple(target.shape) == (10, 10)
''',
    )
    run_notebook(
        'TensorFlow2.4激活函数.ipynb', out,
        assertion='tf.debugging.assert_near(output, tf.constant([-0.02, -0.005, 0.5, 2.]))',
    )
    print('PASS tensorflow: tensor, chronological batches, model shapes and activations', flush=True)


def check_bp(out):
    run_notebook('人工智能基础3.1反向传播算法.ipynb', out, assertion='''
assert len(testY) == 800
assert len(pred) == len(testY)
assert np.isfinite(train_loss_list).all()
assert train_loss_list[-1] < train_loss_list[0]
print('PASS bp: 800 test labels and decreasing training loss')
''')


def check_rnn(out):
    run_notebook(
        '人工智能基础3.2循环神经网络.ipynb', out,
        overrides={28: ('range(200)', 'range(1)')},
        assertion='''
assert tuple(test1.shape) == (269, 1, 5)
assert tuple(test_predictions.shape) == (269, 1, 1)
assert np.isfinite(mse)
np.testing.assert_allclose(mse / float(scale)**2, normalized_test_mse, rtol=1e-5)
print('PASS rnn: 1 epoch, 269 forecasts and correctly scaled MSE')
''',
    )


def check_cnn(out):
    setup = "import os\n" f"os.environ['COURSE_OUTPUT_DIR'] = {str(out / '图形')!r}"
    run_notebook(
        'CNN.ipynb', out,
        overrides={
            29: ('epochs = 20', 'epochs = 1'),
            44: ('for epoch in range(epochs):',
                 'train_loader = train_loader.take(2)\ntest_loader = test_loader.take(2)\nfor epoch in range(1):'),
        },
        setup=setup,
        assertion='''
assert future == 300 and np.isfinite(true_predictions).all()
assert len(train_dataset) == 60000 and len(test_dataset) == 10000
batch, labels = next(iter(test_loader))
assert tuple(batch.shape) == (64, 28, 28, 1)
assert tuple(net(batch, training=False).shape) == (64, 10)
assert np.isfinite(train_loss_list + test_loss_list).all()
assert 0 <= test_acc_list[0] <= 1
print('PASS cnn: 300 recursive forecasts, local MNIST, training and (64, 10) logits')
''',
    )


def graph_check(unit, out):
    import numpy as np

    if unit == 'gcn':
        directory = next(EXTRACTED.glob('人工智能基础4*')) / '3.4.3'
        nodes, first, last = 156, 2616, 2975
        ablation = []
        run_notebook(
            '人工智能基础3.4-图卷积神经网络.ipynb', out,
            source_cells=[4, 7, 10, 13, 16, 18, 20],
            assertion="assert tuple(output.shape) == (4, 3)\nprint('PASS gcn NumPy: (4, 3)')",
        )
    else:
        directory = next(EXTRACTED.glob('案例实战1*')) / 'subway flow prediction_for book'
        nodes, first, last = 276, 1440, 1799
        ablation = ['--inflow-only']

    command = [sys.executable, '-B', '-X', 'utf8']
    training = out / '训练'
    subprocess.run([*command, str(directory/'main.py'), '--epochs', '1', '--patience', '1',
                    '--device', 'cpu', '--output-dir', str(training), *ablation],
                   cwd=ROOT, check=True)
    checkpoint = training / 'best.weights.h5'

    prediction = out / '预测'
    subprocess.run([*command, str(directory/'main_predict.py'), '--checkpoint', str(checkpoint),
                    '--output-dir', str(prediction), '--device', 'cpu'],
                   cwd=ROOT, check=True)
    metrics = json.loads((prediction/'metrics.json').read_text(encoding='utf-8'))
    assert metrics['split_metadata_verified'] and metrics['samples'] == 360
    if unit == 'metro':
        assert metrics['inputs'] == ['inflow']
    with np.load(prediction/'predictions.npz') as data:
        assert data['predicted'].shape == (360, nodes, 1)
        np.testing.assert_array_equal(data['target_index'], np.arange(first, last + 1))
        assert np.isfinite(data['predicted']).all()
    print(f'PASS {unit}: metrics.json, predictions.npz and (360, {nodes}, 1) predictions', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('unit', choices=('env', 'all', *UNITS))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.unit == 'env':
        check_environment()
        return
    output = args.output_dir or COURSE_OUTPUT / '快速验收' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    selected = UNITS if args.unit == 'all' else (args.unit,)
    for unit in selected:
        section = output / unit
        section.mkdir(parents=True, exist_ok=False)
        if unit == 'python':
            check_python()
        elif unit == 'tensorflow':
            check_tensorflow(section)
        elif unit == 'bp':
            check_bp(section)
        elif unit == 'rnn':
            check_rnn(section)
        elif unit == 'cnn':
            check_cnn(section)
        else:
            graph_check(unit, section)
    print(f'Output: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()

