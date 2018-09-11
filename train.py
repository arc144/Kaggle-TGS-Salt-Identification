import torch.optim as optim


def train(net, train_data, val_data, epochs=10):
    num_iters = 300 * 1000
    iter_smooth = 20
    iter_log = 50
    iter_valid = 100

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.005, momentum=0.9, weight_decay=0.0001)
    while iter < num_iters:
        optimizer.zero_grad()
