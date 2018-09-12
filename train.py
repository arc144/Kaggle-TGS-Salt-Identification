import torch.optim as optim
import torch
from contextlib import contextmanager
import numpy as np
import  time
from Analysis import show_image_mask

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def train(net, train_loader, val_loader, lr=0.005, n_epoch=10):
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=lr, momentum=0.9, weight_decay=0.0001)
    net.cuda()

    epoch = 0
    while epoch <= n_epoch:
        with timer('Train Epoch {}'.format(epoch)):
            # Training step
            train_loss, train_iou = training_step(train_loader, optimizer, net)

        #  Validation
        val_loss, val_iou = perform_validation(val_loader, net)

        # Print statistics
        epoch += 1
        print(('train loss: {:.3f}   val_loss: {:.3f}   '
               'train iou:  {:.3f}   val_iou:  {:.3f}').format(
            np.mean(train_loss),
            np.mean(val_loss),
            np.mean(train_iou),
            np.mean(val_iou)))


def training_step(train_loader, optimizer, net, debug=False):
    net.set_mode('train')
    train_loss = []
    train_iou = []
    for index, im, mask in train_loader:
        optimizer.zero_grad()
        if debug:
            show_image_mask(im.numpy(), mask.numpy())
        im = im.cuda()
        mask = mask.cuda()
        logit = net(im)
        loss = net.criterion(logit, mask)
        iou  = net.metric(logit, mask)
        train_loss.append(loss.item())
        train_iou.append(iou.item())
        loss.backward()
        optimizer.step()
    return train_loss, train_iou


def perform_validation(val_loader, net):
    net.set_mode('valid')
    count = 0
    val_loss = []
    val_iou = []
    for index, im, mask in val_loader:
        im = im.cuda()
        mask = mask.cuda()
        with torch.no_grad():
            logit = net(im)
            loss = net.criterion(logit, mask)
            iou  = net.metric(logit, mask)
        val_loss.append(loss.item())
        val_iou.append(iou.item())
    return val_loss, val_iou
