from Dataset import TGS_Dataset
from Models import UNetResNet34, UNetResNet34_convT, UNetResNet34_convT_hyper
from contextlib import contextmanager
import time
import os

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

##############################
TRAIN_PATH = './Data/Train'
TEST_PATH = './Data/Test'
PREDICT_TEST = True
LOAD_PATH = None #'./Saves/UNetResNet34_convT/2018-09-14 10:43_Epoach27_val0.806'
OVERWRITE_SAVE = False
DEBUG = False
##############################
LOSS = 'BCE+Dice'
OPTIMIZER = 'SGD'
PRETRAINED = True
BATCH_SIZE = 32
LR = 0.01
GAMMA = 0.25
##############################
THRESHOLD = 0.5
##############################

train_dataset = TGS_Dataset(TRAIN_PATH)
train_dataset.load_images()
loaders, ids = train_dataset.yield_dataloader(num_workers=11, batch_size=BATCH_SIZE)

if PREDICT_TEST:
    test_dataset = TGS_Dataset(TEST_PATH)
    test_dataset.load_images(data='test')
    test_loader, test_ids = test_dataset.yield_dataloader(data='test', num_workers=11, batch_size=BATCH_SIZE)

for i, (train_loader, val_loader) in enumerate(loaders):
    with timer('Fold {}'.format(i)):
        net = UNetResNet34_convT(lr=LR, debug=DEBUG, pretrained=PRETRAINED,
                                       overwrite=OVERWRITE_SAVE if not i else False)
        net.define_criterion(LOSS)
        net.create_optmizer(optimizer=OPTIMIZER, use_scheduler=True, gamma=GAMMA)
        if LOAD_PATH is not None:
            net.load_model(LOAD_PATH)
        net.train_network(train_loader, val_loader, n_epoch=40)
        net.plot_training_curve(show=True)
        if PREDICT_TEST:
            df = net.predit(test_loader, threshold=THRESHOLD)
            df.to_csv(os.path.join('./Saves', type(net).__name__), index=False)


