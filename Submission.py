import gc
import os
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from Dataset import TGS_Dataset
from Evaluation import do_length_decode, do_length_encode
from Models import UNetResNet34_SE, UNetResNet34_SE_Hyper, UNetResNet34_SE_Hyper_v2
from Augmentation import do_horizontal_flip

# UNTESTED, NEED TO COMPARE WITH NATIVE MEAN
def average_fold_predictions(path_list, H=101, W=101, fill_value=255, threshold=0.5):
    '''Load rle from df, average them and return a new rle df'''
    folds = []
    # decode
    for p in path_list:
        df = pd.read_csv(p)
        im = []
        for i in range(len(df)):
            im.append(do_length_decode(str(df.rle_mask.iloc[i]), H, W, fill_value))
        folds.append(im)
    # average
    avg = np.mean(folds, axis=0)
    avg = avg > threshold
    # encode
    rle = []
    for i in range(len(avg)):
        rle.append(do_length_encode(avg[i]))
    # create sub
    df = pd.DataFrame(dict(id=df.id, rle_mask=rle))
    return df

def load_net_and_predict(net, test_path, load_paths, batch_size=32, tta_transform=None, threshold=0.5):
    test_dataset = TGS_Dataset(test_path)
    # test_dataset.load_images(data='test')
    test_loader, test_ids = test_dataset.yield_dataloader(data='test', num_workers=11, batch_size=batch_size)
    # predict
    for i in tqdm(range(len(load_paths))):
        net.load_model(load_paths[i])
        p = net.predict(test_loader, threshold=0, tta_transform=tta_transform, return_rle=False)
        if not i:
            avg = np.zeros_like(p['pred'])

        avg = (i * avg + p['pred']) / (i + 1)

    avg = avg > threshold
    # free some memory
    del test_dataset, test_loader
    gc.collect()
    # encode
    rle = []
    for i in range(len(avg)):
        rle.append(do_length_encode(avg[i]))
    # create sub
    df = pd.DataFrame(dict(id=p['id'], rle_mask=rle))
    return df

def tta_transform(images, mode):
    out = []
    if mode == 'out':
        images = images[0]
    images = images.transpose((0, 2, 3, 1))
    tta = []
    for i in range(len(images)):
        t = np.fliplr(images[i])
        tta.append(t)
    tta = np.transpose(tta, (0, 3, 1, 2))
    out.append(tta)
    return np.asarray(out)

if __name__ == '__main__':
    TEST_PATH = './Data/Test'
    DEBUG = False
    net = UNetResNet34_SE_Hyper_v2(debug=DEBUG)
    NET_NAME = type(net).__name__
    THRESHOLD = 0.45
    BATCH_SIZE = 32

    LOAD_PATHS = [
        './Saves/UNetResNet34_SE_Hyper_v2/2018-09-20 09:25_Fold1_Epoach76_reset1_val0.844',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 01:05_Fold1_Epoach69_reset1_val0.840',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 01:30_Fold1_Epoach103_reset2_val0.847',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 01:53_Fold1_Epoach135_reset3_val0.841',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 02:33_Fold2_Epoach30_reset0_val0.852',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 03:05_Fold2_Epoach74_reset1_val0.853',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 03:35_Fold2_Epoach115_reset2_val0.856',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 04:07_Fold2_Epoach158_reset3_val0.858',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 04:33_Fold3_Epoach34_reset0_val0.841',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 04:56_Fold3_Epoach66_reset1_val0.839',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 05:26_Fold3_Epoach107_reset2_val0.833',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 05:59_Fold3_Epoach152_reset3_val0.837',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 06:30_Fold4_Epoach35_reset0_val0.829',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 06:59_Fold4_Epoach74_reset1_val0.833',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 07:27_Fold4_Epoach113_reset2_val0.833',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 07:59_Fold4_Epoach156_reset3_val0.828',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 08:25_Fold5_Epoach32_reset0_val0.858',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 08:48_Fold5_Epoach63_reset1_val0.860',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 09:26_Fold5_Epoach113_reset2_val0.856',
        # './Saves/UNetResNet34_SE_Hyper/2018-09-19 09:49_Fold5_Epoach144_reset3_val0.859',
    ]

    ################################################
    # df = average_fold_predictions(SUB_PATHS)
    df = load_net_and_predict(net, TEST_PATH, LOAD_PATHS,
                              # tta_transform=tta_transform,
                              batch_size=BATCH_SIZE,
                              threshold=THRESHOLD)
    # SUB_PATHS = ['/media/data/Kaggle/Kaggle-TGS-Salt-Identification/Saves/UNetResNet34_SE_PPM/sub_fold0_val0.792.csv']
    df.to_csv(os.path.join(
        './Saves',
        NET_NAME,
        '{}_5foldAvg.csv'.format(NET_NAME)),
        index=False)