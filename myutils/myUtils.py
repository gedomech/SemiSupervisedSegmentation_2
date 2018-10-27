import numpy as np
import torch
import numpy as np,pandas as pd, matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

def pred2segmentation(prediction):
    return prediction.max(1)[1]


def dice_loss(input, target):
    # with torch.no_grad:
    smooth = 1.

    iflat = input.view(input.size(0),-1)
    tflat = target.view(input.size(0),-1)
    intersection = (iflat * tflat).sum(1)

    return ((2. * intersection + smooth).float() /  (iflat.sum(1) + tflat.sum(1) + smooth).float()).mean()

def iou_loss(pred, target, n_class):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            try:
                ious.append(float(intersection) / max(union, 1).cpu().data.numpy())
            except:
                ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def image_batch_generator(dataset=None, batch_size=1, number_workers=1, device=torch.device):
    """
    This function generates batches containing (images, masks, paths)
    :param dataset: torch.utils.data.Dataset object to be loaded
    :param batch_size: size of the batch
    :param number_workers: number of threads used to load data
    :param device: torch.device object where images and masks will be located.
    :return: (images, masks, paths)
    """
    if not issubclass(type(dataset), torch.utils.data.Dataset):
        raise TypeError("Input must be an instance of the torch.utils.data.Dataset class")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=number_workers, pin_memory=True)

    try:
        _, data_batch = enumerate(data_loader).__next__()
    except:
        labeled_loader_iter = enumerate(data_loader)
        _, data_batch = labeled_loader_iter.__next__()
    img, mask, paths = data_batch
    return img.to(device), mask.to(device), paths


def save_models(nets, score=None, epoch=0):
    """

    :param nets:
    :param score:
    :param epoch:
    :return:
    """
    pass


class Colorize:

    def __init__(self, n=4):
        self.cmap = colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.squeeze().size()
            # size = gray_image.squeeze().size()
        try:
            color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        except:
            color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)



        for label in range(1, len(self.cmap)):
            mask = gray_image.squeeze() == label
            try:
                color_image[0][mask] = self.cmap[label][0]
                color_image[1][mask] = self.cmap[label][1]
                color_image[2][mask] = self.cmap[label][2]
            except:
                print('error in colorize.')
        return color_image

def showImages(board,image_batch, mask_batch,segment_batch):
    color_transform = Colorize()
    means = np.array([0.762824821091, 0.546326646928, 0.570878231817])
    stds = np.array([0.0985789149783, 0.0857434017536, 0.0947628491147])
    # import ipdb
    # ipdb.set_trace()
    if image_batch.min()<0:
        for i in range(3):
            image_batch[:,i,:,:]=(image_batch[:,i,:,:])*stds[i]+means[i]


    board.image(image_batch[0], 'original image')
    board.image(color_transform(mask_batch[0]), 'ground truth image')
    board.image(color_transform(segment_batch[0]), 'prediction given by the net')