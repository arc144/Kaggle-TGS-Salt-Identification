import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from Augmentation import *

IM_SIZE = 101


def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i

def normalize(im):
    max = np.max(im)
    min = np.min(im)
    if (max - min) > 0:
        im = (im - min) / (max - min)
    return im


def basic_augment(image, mask, index):
    if np.random.rand() < 0.5:
        image, mask = do_horizontal_flip2(image, mask)
        pass

    if np.random.rand() < 0.5:
        c = np.random.choice(4)
        if c == 0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.2)  # 0.125

        if c == 1:
            image, mask = do_horizontal_shear2(image, mask, dx=np.random.uniform(-0.07, 0.07))
            pass

        if c == 2:
            image, mask = do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0, 15))  # 10

        if c == 3:
            image, mask = do_elastic_transform2(image, mask, grid=10, distort=np.random.uniform(0, 0.15))  # 0.10
            pass

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c == 0:
            image = do_brightness_shift(image, np.random.uniform(-0.1, +0.1))
        if c == 1:
            image = do_brightness_multiply(image, np.random.uniform(1 - 0.08, 1 + 0.08))
        if c == 2:
            image = do_gamma(image, np.random.uniform(1 - 0.08, 1 + 0.08))
        # if c==1:
        #     image = do_invert_intensity(image)

    return image, mask, index


class TorchDataset(Dataset):

    def __init__(self, df, test=False, transform=None):
        self.df = df
        self.test = test
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        pad = ((0, 0), (14, 13), (14, 13))

        im = self.df.images.iloc[index]

        if self.test:
            mask = None
        else:
            mask = self.df.masks.iloc[index]

            if self.transform is not None:
                im, mask, index = self.transform(im, mask, index)

            mask = np.expand_dims(mask, 0)
            mask = np.pad(mask, pad, 'reflect')
            mask = torch.from_numpy(mask).float()


        im = np.rollaxis(im, 2, 0)
        # im = np.expand_dims(im, 0)
        im = np.pad(im, pad, 'reflect')
        im = torch.from_numpy(im).float()

        return index, im, mask


class TGS_Dataset():

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.df = self.create_dataset_df(self.folder_path)

    def load_images(self, data='train'):
        self.df['images'] = [
            cv2.imread(self.df.loc[i]['im_path'],
                       cv2.IMREAD_COLOR).astype(np.float32) / 255 for i in self.df.index]
        if data == 'train':
            self.df['masks'] = [
                cv2.imread(self.df.loc[i]['mask_path'],
                           cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 for i in self.df.index]

    @staticmethod
    def create_dataset_df(folder_path):
        '''Create a dataset for a specific dataset folder path'''
        # Walk and get paths
        walk = os.walk(folder_path)
        main_dir_path, subdirs_path, rle_path = next(walk)
        dir_im_path, _, im_path = next(walk)
        # Create dataframe
        df = pd.DataFrame()
        df['id'] = [im_p.split('.')[0] for im_p in im_path]
        df['im_path'] = [os.path.join(dir_im_path, im_p) for im_p in im_path]
        if any(['mask' in sub for sub in subdirs_path]):
            dir_mask_path, _, mask_path = next(walk)
            df['mask_path'] = [os.path.join(dir_mask_path, m_p)
                               for m_p in mask_path]
            rle_df = pd.read_csv(os.path.join(main_dir_path, rle_path[0]))
            df = df.merge(rle_df, on='id', how='left')

        return df

    def yield_dataloader(self, data='train', nfold=5,
                         shuffle=True, seed=143, stratify=True,
                         num_workers=8, batch_size=10):

        if data == 'train':
            if stratify:
                self.df["coverage"] = self.df.masks.map(np.sum) / pow(IM_SIZE, 2)
                self.df["coverage_class"] = self.df.coverage.map(cov_to_class)
                kf = StratifiedKFold(n_splits=nfold,
                                     shuffle=True,
                                     random_state=seed)
            else:
                kf = KFold(n_splits=nfold,
                           shuffle=True,
                           random_state=seed)
            loaders = []
            idx = []
            for train_ids, val_ids in kf.split(self.df['id'].values, self.df.coverage_class):
                train_dataset = TorchDataset(self.df.iloc[train_ids],
                                             transform=basic_augment)
                train_loader = DataLoader(train_dataset,
                                          shuffle=shuffle,
                                          num_workers=num_workers,
                                          batch_size=batch_size,
                                          pin_memory=True)

                val_dataset = TorchDataset(self.df.iloc[val_ids])
                val_loader = DataLoader(val_dataset,
                                        shuffle=shuffle,
                                        num_workers=num_workers,
                                        batch_size=batch_size,
                                        pin_memory=True)
                idx.append((self.df.id.iloc[train_ids], self.df.id.iloc[val_ids]))
                loaders.append((train_loader, val_loader))
            return loaders, idx

        elif data == 'test':
            test_dataset = TorchDataset(self.df)
            test_loader = DataLoader(test_dataset,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     batch_size=batch_size,
                                     pin_memory=True)
            return test_loader, self.df.id

    def visualize_sample(self, sample_size):
        samples = np.random.choice(self.df['id'].values, sample_size)
        self.df.set_index('id', inplace=True)
        fig, axs = plt.subplots(2, sample_size)
        for i in range(sample_size):
            im = cv2.imread(self.df.loc[samples[i], 'im_path'], cv2.IMREAD_COLOR)
            mask = cv2.imread(self.df.loc[samples[i], 'mask_path'], cv2.IMREAD_GRAYSCALE)
            print('Image shape: ', np.array(im).shape)
            print('Mask shape: ', np.array(mask).shape)
            axs[0, i].imshow(im)
            axs[1, i].imshow(mask)


if __name__ == '__main__':
    TRAIN_PATH = './Data/Train'
    TEST_PATH = './Data/Test'

    dataset = TGS_Dataset(TRAIN_PATH)
    # dataset.visualize_sample(3)
    loaders, idx = dataset.yield_dataloader(data='train', nfold=5,
                                            shuffle=True, seed=143,
                                            num_workers=8, batch_size=10)
    ids = []
    for index, im, mask in loaders[0][0]:
        ids.append(index)

    print(len(ids))
    # plt.show()
