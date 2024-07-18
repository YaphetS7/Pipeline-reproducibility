import os

path2save = 'runs/2'

os.makedirs(path2save, exist_ok=True)

SEED = 42
LOADER_SEED = 20

batch_size = 16
epochs = 10