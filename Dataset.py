import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

def normalize(im):
    if np.max(im):
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
    return im


class TorchDataset(Dataset):

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        im = Image.open(self.df.iloc[index]['im_path'])
        mask = Image.open(self.df.iloc[index]['mask_path'])

        if self.transform is not None:
            im, mask = self.transform(im, mask)

        # Convert image and label to torch tensors
        im = np.asarray(im)
        pad = ((14, 13), (14, 13), (0, 0))
        im = np.pad(im, pad, 'reflect').transpose((2, 0, 1))
        im = normalize(im)
        im = torch.from_numpy(im).float()

        mask = np.asarray(mask)
        mask = np.expand_dims(mask, -1)
        mask = np.pad(mask, pad, 'reflect').transpose((2, 0, 1))
        mask = normalize(mask)
        mask = torch.from_numpy(mask).float()

        return index, im, mask


class TGS_Dataset():

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.df = self.create_dataset_df(self.folder_path)

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

    def yield_dataloader(self, data='train', split_method='kfold', nfold=5,
                         shuffle=True, seed=143,
                         num_workers=8, batch_size=10):
        if data == 'train':
            if split_method == 'kfold':
                kf = KFold(n_splits=nfold,
                           shuffle=True,
                           random_state=seed)
                loaders = []
                idx = []
                for train_ids, val_ids in kf.split(self.df['id'].values):
                    train_loader = DataLoader(TorchDataset(self.df.iloc[
                                                               train_ids]),
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              pin_memory=True)
                    val_loader = DataLoader(TorchDataset(self.df.iloc[
                                                             val_ids]),
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            batch_size=batch_size,
                                            pin_memory=True)
                    idx.append((train_ids, val_ids))
                    loaders.append((train_loader, val_loader))
        # elif data == 'test':
        return loaders, idx

    def visualize_sample(self, sample_size):
        samples = np.random.choice(self.df['id'].values, sample_size)
        self.df.set_index('id', inplace=True)
        fig, axs = plt.subplots(2, sample_size)
        for i in range(sample_size):
            im = Image.open(self.df.loc[samples[i], 'im_path'])
            mask = Image.open(self.df.loc[samples[i], 'mask_path'])
            print('Image shape: ', np.array(im).shape)
            print('Mask shape: ', np.array(mask).shape)
            axs[0, i].imshow(im)
            axs[1, i].imshow(mask)


if __name__ == '__main__':
    TRAIN_PATH = './Data/Train'
    TEST_PATH = './Data/Test'

    dataset = TGS_Dataset(TRAIN_PATH)
    # dataset.visualize_sample(3)
    loaders, idx = dataset.yield_dataloader(data='train', split_method='kfold', nfold=5,
                                            shuffle=True, seed=143,
                                            num_workers=8, batch_size=10)
    ids = []
    for index, im, mask in loaders[0][0]:
        ids.append(index)

    print(len(ids))
    # plt.show()
