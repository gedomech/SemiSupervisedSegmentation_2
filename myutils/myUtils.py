import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from myutils.myLoss import JensenShannonDivergence


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


def image_batch_generator(dataset=None, device=torch.device):
    """
    This function generates batches containing (images, masks, paths)
    :param dataset: torch.utils.data.Dataset object to be loaded
    :param batch_size: size of the batch
    :param number_workers: number of threads used to load data
    :param device: torch.device object where images and masks will be located.
    :return: (images, masks, paths)
    """
    if not issubclass(type(dataset), DataLoader):
        raise TypeError("Input must be an instance of the torch.utils.data.Dataset class")


    try:
        _, data_batch = enumerate(dataset).__next__()
    except:
        labeled_loader_iter = enumerate(dataset)
        _, data_batch = labeled_loader_iter.__next__()
    img, mask, paths = data_batch
    return img.to(device), mask.to(device), paths


def save_models(nets_, nets_path_, nets_names, score_meters=None, epoch=0, history_score_dict={}):
    """
    This function saves the parameters of the nets
    :param nets_: networks containing the parameters to be saved
    :param nets_path_: list of path where each net will be saved
    :param nets_names: list of names to recovery nets from dictionary
    :param score_meters: list of torchnet.meter.AverageValueMeter objects corresponding with each net
    :param epoch: epoch which was obtained the scores
    :param history_score_dict:
    :return:
    """
    history_score_dict['epoch']=epoch+1

    for idx, net_i in enumerate(nets_):

        if (idx == 0) and ( history_score_dict[nets_names[idx]] < score_meters[idx].value()[0]):
            history_score_dict[nets_names[idx]] = score_meters[idx].value()[0]
            print('The highest dice score for {} is {:.3f} in the test'.format(nets_names[idx],
                                                                               history_score_dict[nets_names[idx]]))
            torch.save(net_i.state_dict(), nets_path_[idx])

        elif (idx == 1) and (history_score_dict[nets_names[idx]] < score_meters[idx].value()[0]):
            history_score_dict[nets_names[idx]] = score_meters[idx].value()[0]
            print('The highest dice score for {} is {:.3f} in the test'.format(nets_names[idx],
                                                                               history_score_dict[nets_names[idx]]))
            torch.save(net_i.state_dict(), nets_path_[idx])

        elif (idx == 2) and (history_score_dict[nets_names[idx]] < score_meters[idx].value()[0]):
            history_score_dict[nets_names[idx]]= score_meters[idx].value()[0]
            print('The highest dice score for {} is {:.3f} in the test'.format(nets_names[idx],
                                                                               history_score_dict[nets_names[idx]]))
            torch.save(net_i.state_dict(), nets_path_[idx])

    return history_score_dict


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


def learning_rate_decay(optims, factor=0.95):
    for opti_i in optims:
        for param_group in opti_i.param_groups:
            param_group['lr'] = param_group['lr'] * factor


def map_(func,*list):
    return [*map(func,*list)]


def batch_labeled_loss_(img,mask,nets,criterion):
    loss_list = []
    prediction_list = []
    dice_score = []
    for net_i in nets:
        pred = net_i(img)
        labeled_loss = criterion(pred, mask.squeeze(1))
        loss_list.append(labeled_loss)
        ds = dice_loss(pred2segmentation(net_i(img)), mask.squeeze(1))
        dice_score.append(ds)
        prediction_list.append(pred)

    return prediction_list, loss_list, dice_score


def batch_labeled_loss_customized(labeled_loaders, device_, nets, criterion):
    loss_list = []
    prediction_list = []
    dice_score = []
    for idx, loader in enumerate(labeled_loaders):
        img, mask, _ = image_batch_generator(loader, device=device_)
        pred = nets[idx](img)
        labeled_loss = criterion(pred, mask.squeeze(1))
        loss_list.append(labeled_loss)
        ds = dice_loss(pred2segmentation(pred), mask.squeeze(1))
        dice_score.append(ds)
        prediction_list.append(pred)

    return prediction_list, loss_list, dice_score


def batch_labeled_loss_custom(img_list, mask_list, nets, criterion):
    """
    This function compute the loss for each net from the customized training sets
    :param img_list:
    :param mask_list:
    :param nets:
    :param criterion:
    :return:
    """
    loss_list = []
    prediction_list = []
    dice_score = []
    for idx, net_i in enumerate(nets):
        pred_i = net_i(img_list[idx])
        labeled_loss = criterion(pred_i, mask_list[idx].squeeze(1))
        loss_list.append(labeled_loss)
        ds = dice_loss(pred2segmentation(net_i(img_list[idx])), mask_list[idx].squeeze(1))
        dice_score.append(ds)
        prediction_list.append(pred_i)

    return prediction_list, loss_list, dice_score


from torchnet.meter import AverageValueMeter
import torch.nn.functional as F


def test(nets_,  test_loader_,device, **kwargs):
    class_number =2

    """
    This function performs the evaluation with the test set containing labeled images.
    """

    map_(lambda x:x.eval(), nets_)

    dice_meters_test = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]
    mv_dice_score_meter = AverageValueMeter()

    with torch.no_grad():
        for i,(img, mask , _) in enumerate(test_loader_):

            (img, mask) = img.to(device), mask.to(device)
            distributions = torch.zeros([img.shape[0], class_number, img.shape[2], img.shape[3]]).to(device)

            for idx, net_i in enumerate(nets_):
                pred_test = nets_[idx](img)
                # plt.imshow(pred_test[0, 1].cpu().numpy())

                distributions += F.softmax(pred_test, 1)
                dice_test = dice_loss(pred2segmentation(pred_test), mask.squeeze(1))
                dice_meters_test[idx].add(dice_test)

                # To test validation value per net

                print('For image {} dice_meters_test value {} for net {}'.format(i,
                                                                                 dice_test,
                                                                                 idx))

            distributions /= len(nets_)
            mv_dice_score = dice_loss(pred2segmentation(distributions), mask.squeeze(1))
            mv_dice_score_meter.add(mv_dice_score.item())

    map_(lambda x:x.train(), nets_)

    return [dice_meters_test[idx] for idx in range(len(nets_))], mv_dice_score_meter


def get_mv_based_labels(imgs, nets):
    class_number =2
    prediction = []
    distributions = torch.zeros([imgs.shape[0], class_number, imgs.shape[2], imgs.shape[3]]).to(imgs.dtype)
    for idx, (net_i) in enumerate(nets):
        pred = F.softmax(net_i(imgs))
        prediction.append(pred)
        distributions+=pred.cpu()
    distributions /= len(nets)
    return pred2segmentation(distributions), prediction


def cotraining(prediction, pseudolabel, nets,criterion, device):
    loss = []
    for idx, net_i in enumerate(nets):
        unlabled_loss = criterion(prediction[idx], pseudolabel.to(device))
        loss.append(unlabled_loss)
    return loss


def get_loss(predictions):
    p = torch.cat(predictions)
    criteron = JensenShannonDivergence()
    loss = criteron(p)
    return loss


def visualize(writer, nets_, image_set, n_images, c_epoch, randomly=True,  nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """
    Visualize n_images from the input set of images (image_set).
    :param nets_: networks used to extract the predictions from the input images
    :param image_set: set of images to be visualized
    :param n_images: number of images that really will be visualized
    :param c_epoch: current epoch
    :param randomly: indicates if n_images will be randomly taken from image_set

    The rest of parameters correspond to the input arguments of torchvision.utils.make_grid.
    For more documentation refers to https://pytorch.org/docs/stable/torchvision/utils.html
    :param nrow:
    :param padding:
    :param normalize:
    :param range:
    :param scale_each:
    :param pad_value:
    :return:
    """
    n_samples = np.min([image_set.shape[0], n_images])

    if randomly:
        idx = np.random.randint(low=0, high=image_set.shape[0], size=n_samples)
    else:
        idx = np.arange(n_samples)

    imgs = image_set[idx, :, :, :]
    for idx, net_i in enumerate(nets_):
        pred_grid = vutils.make_grid(net_i(imgs).cpu(), nrow=nrow, padding=padding, pad_value=pad_value,
                                     normalize=normalize, range=range, scale_each=scale_each)
        if idx == 0:
            writer.add_image('Enet Predictions', pred_grid, c_epoch)  # Tensor
        elif idx == 1:
            writer.add_image('Unet Predictions', pred_grid, c_epoch)  # Tensor
        else:
            writer.add_image('SegNet Predictions', pred_grid, c_epoch)  # Tensor


def add_visual_perform(writer: SummaryWriter, score_meters: dict, c_epoch):

    writer.add_scalars('data/performance_plot', score_meters, c_epoch)


import time
def s_forward_backward(net,optim, imgs, masks, criterion):

    now = time.time()
    optim.zero_grad()
    pred = net(imgs)
    loss = criterion(pred, masks.squeeze(1))
    loss.backward()
    optim.step()
    dice_score = dice_loss(pred2segmentation(pred), masks.squeeze(1))

    return dice_score


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

