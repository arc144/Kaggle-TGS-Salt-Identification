from Dataset import TGS_Dataset
from Model import UNetResNet34

TRAIN_PATH = './Data/Train'
TEST_PATH = './Data/Test'

train_dataset = TGS_Dataset(TRAIN_PATH)
loaders = train_dataset.yield_dataloader()

net = UNetResNet34()

for i, (train_loader, val_loader) in enumerate(loaders):
    print('Starting fold {} out of {}'.format(i, len(loaders)))
    if i:
        pass
