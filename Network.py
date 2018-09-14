import torch.optim as optim
import torch.nn as nn
import torch

from contextlib import contextmanager
import datetime
import  time
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Loss
from Analysis import show_image_mask, show_image_mask_pred
from Evaluation import  do_kaggle_metric, dice_accuracy, do_mAP, batch_encode

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.3f}s".format(title, time.time() - t0))


class SegmentationNetwork(nn.Module):

    def __init__(self, lr=0.005, overwrite=True, debug=False):
        super(SegmentationNetwork, self).__init__()
        self.lr = lr
        self.overwrite = overwrite
        self.debug = debug
        self.scheduler = None
        self.best_metric = 0
        self.epoch = 0

        self.train_log = dict(loss=[], iou=[], mAP=[])
        self.val_log = dict(loss=[], iou=[], mAP=[])
        self.create_save_folder()

    def create_optmizer(self, optimizer='SGD', use_scheduler=False, gamma=0.25):
        self.cuda()
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                  lr=self.lr, momentum=0.9, weight_decay=0.0001)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                       self.parameters()), lr=self.lr)

        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                  mode='max',
                                                                  factor=gamma,
                                                                  patience=4,
                                                                  verbose=True,
                                                                  threshold=0.01,
                                                                  min_lr=1e-05,
                                                                  eps=1e-08)


    def train_network(self, train_loader, val_loader, n_epoch=10):
        while self.epoch < n_epoch:
            self.epoch += 1
            with timer('Train Epoch {}/{}'.format(self.epoch, n_epoch)):
                # Training step
                train_loss, train_iou, train_mAP = self.training_step(train_loader)
                #  Validation
                val_loss, val_iou, val_mAP = self.perform_validation(val_loader)
                # Learning Rate Scheduler
                if self.scheduler is not None:
                    self.scheduler.step(np.mean(val_mAP))
                # Save best model
                self.save_best_model(np.mean(val_mAP))


            # Print statistics
            print(('train loss: {:.3f}  val_loss: {:.3f}  '
                   'train iou:  {:.3f}  val_iou:  {:.3f}  '
                   'train mAP:  {:.3f}  val_mAP:  {:.3f}').format(
                np.mean(train_loss),
                np.mean(val_loss),
                np.mean(train_iou),
                np.mean(val_iou),
                np.mean(train_mAP),
                np.mean(val_mAP)))


    def training_step(self, train_loader):
        self.set_mode('train')
        train_loss = []
        train_iou = []
        train_mAP = []
        for i, (index, im, mask) in enumerate(train_loader):
            self.optimizer.zero_grad()
            im = im.cuda()
            mask = mask.cuda()
            logit = self.forward(im)
            pred = torch.sigmoid(logit)

            loss = self.criterion(logit, mask)
            iou  = dice_accuracy(pred, mask, is_average=False)
            mAP = do_mAP(pred.data.cpu().numpy(), mask.cpu().numpy(), is_average=False)

            train_loss.append(loss.item())
            train_iou.extend(iou)
            train_mAP.extend(mAP)

            loss.backward()
            self.optimizer.step()

            if self.debug and not self.epoch % 5 and not i % 30:
                show_image_mask_pred(
                    im.cpu().data.numpy(), mask.cpu().data.numpy(), logit.cpu().data.numpy())
        # Append epoch data to metrics dict
        for metric in ['loss', 'iou', 'mAP']:
            self.train_log[metric].append(np.mean(eval('train_{}'.format(metric))))
        return train_loss, train_iou, train_mAP


    def perform_validation(self, val_loader):
        self.set_mode('valid')
        val_loss = []
        val_iou = []
        val_mAP = []
        for index, im, mask in val_loader:
            im = im.cuda()
            mask = mask.cuda()

            with torch.no_grad():
                logit = self.forward(im)
                pred = torch.sigmoid(logit)
                loss = self.criterion(logit, mask)
                iou  = dice_accuracy(pred, mask, is_average=False)
                mAP = do_mAP(pred.cpu().numpy(), mask.cpu().numpy(), is_average=False)

            val_loss.append(loss.item())
            val_iou.extend(iou)
            val_mAP.extend(mAP)
        # Append epoch data to metrics dict
        for metric in ['loss', 'iou', 'mAP']:
            self.val_log[metric].append(np.mean(eval('val_{}'.format(metric))))

        return val_loss, val_iou, val_mAP


    def predit(self, test_loader, threshold=0.5):
        self.set_mode('test')
        ids = []
        rle = []
        for i, (idx, im, _) in enumerate(test_loader):
            im.cuda()
            with torch.no_grad():
                logit = self.forward(im)
                pred = logit > threshold
                rle.extend(batch_encode(pred.data.numpy()))
                ids.extend(idx)
        return pd.DataFrame(dict(id=ids, rle_mask=rle))


    def define_criterion(self, name):
        if name == 'BCE+Dice':
            self.criterion = Loss.BCE_Dice()
        elif name == 'Dice':
            self.criterion = Loss.DiceLoss()
        elif name == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        elif name == 'RobustFocal':
            self.criterion = Loss.RobustFocalLoss2d()


    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


    def save_best_model(self, metric, mode='max'):
        if (mode == 'max' and metric > self.best_metric) or (mode == 'min' and metric < self.best_metric):
            self.best_metric = metric
            date = ':'.join(str(datetime.datetime.now()).split(':')[:2])
            p = os.path.join(self.save_dir, '{:}_Epoach{}_val{:.3f}'.format(date, self.epoch, metric))
            torch.save(self.state_dict(), p)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def create_save_folder(self):
        name = type(self).__name__
        self.save_dir = os.path.join('./Saves', name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        elif self.overwrite:
            shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir)

    def plot_training_curve(self, show=True):
        fig, axs = plt.subplots(1, 3)
        for i, metric in enumerate(['loss', 'iou', 'mAP']):
            axs[i].plot(self.train_log[metric], 'ro-', label='Train')
            axs[i].plot(self.val_log[metric], 'bo-', label='Validation')
            axs[i].legend()
            axs[i].set_title(metric)
            axs[i].set_xlabel('Epochs')
            axs[i].set_ylabel(metric)
        if show:
            plt.show()
