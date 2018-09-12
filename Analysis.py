import matplotlib.pyplot as plt
import numpy as np

def show_image_mask(im, mask, n=3, label='Image', show=True, cmap='jet', format='channels_first'):
    im = np.squeeze(im)
    mask = np.squeeze(mask)
    if format == 'channels_first':
        im = im.transpose((0, 2, 3, 1))

    n_batch = im.shape[0]
    idx = np.random.choice(np.arange(n_batch), n, replace=False)
    fig, axs = plt.subplots(2, n)
    for i in range(n):
        axs[0, i].imshow(im[idx[i], :, :, :], cmap=cmap)
        axs[0, i].set_title('{}: {}'.format(label, i))
        axs[1, i].imshow(mask[idx[i], :, :], cmap=cmap)
        axs[1, i].set_title('mask: {}'.format(i))
    if show:
        plt.show()