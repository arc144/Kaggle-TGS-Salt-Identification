from Dataset import TGS_Dataset
from Model import UNetResNet34
from contextlib import contextmanager
from train import train
import time

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

TRAIN_PATH = './Data/Train'
TEST_PATH = './Data/Test'

train_dataset = TGS_Dataset(TRAIN_PATH)
loaders, ids = train_dataset.yield_dataloader()

net = UNetResNet34()

for i, (train_loader, val_loader) in enumerate(loaders):
    with timer('Fold {}'.format(i)):
        if not i:
            train(net, train_loader, val_loader, lr=0.005, n_epoch=20)

