import os

# Pretrained models and reference data live in the repository's top-level
# ``data/`` directory. Set TINNET_DATA_DIR to point somewhere else.
DATA_DIR = os.environ.get(
    'TINNET_DATA_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))


def data_path(*parts):
    return os.path.join(DATA_DIR, *parts)
