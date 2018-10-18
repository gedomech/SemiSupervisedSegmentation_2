import sys, os

sys.path.extend([os.path.dirname(os.getcwd())])
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from myutils.myDataLoader import ISICdata
from myutils.myENet import Enet
from myutils.myNetworks import UNet, SegNet
from myutils.tiramisu import FCDenseNet67
from myutils.myLoss import CrossEntropyLoss2d, JensenShannonDivergence

from tqdm import tqdm
from torchnet.meter import AverageValueMeter
from myutils.myUtils import pred2segmentation, iou_loss, showImages
from myutils.myVisualize import Dashboard
import platform
import torch.backends.cudnn as cudnn


# Jensen-Shannon Divergence
def JSD(prob_dists, _weights=None, logbase=2):
    """
    This function computes the Jense-Shannon Divergence: H(sum(w_i*P_i)) - sum(w_i*H(P_i)).

    :param prob_dists: The distributions, P_i, to take the Jensen-Shannon Divergence of.
    :param _weights: The weights, w_i, to give the distributions. If None, the weights are assumed to be uniform.
    :param logbase: The logarithmic base to use, defaults to 2.
    :return: jsd (jensen-shannon divergence)
    """
    if _weights is None:
        nProbs = len(prob_dists)
        weights = np.empty(nProbs)
        weights.fill(1 / nProbs)

    # left term: entropy of misture
    wprobs = weights * prob_dists
    mixture = wprobs.sum(axis=0)
    entropy_of_mixture = H(mixture, base=logbase)

    print("My entropy", entropy_of_mixture)

    # right term: sum of entropies
    entropies = np.array([H(P_i, base=logbase) for P_i in prob_dists])
    wentropies = weights * entropies
    sum_of_entropies = wentropies.sum()

    jsd = entropy_of_mixture - sum_of_entropies
    return jsd


cuda_device = "0"
root = "datasets/ISIC2018"

class_number = 2
lr = 1e-4
weigth_decay = 1e-6
use_cuda = False  # True
number_workers = 1  # 7
batch_size = 1  # 7
max_epoch = 1  # 100
train_print_frequncy = 1  # 10
val_print_frequncy = 1  # 10
## visualization
board_train_image = Dashboard(server='http://localhost', env="image_train")
board_test_image = Dashboard(server='http://localhost', env="image_test")
board_loss = Dashboard(server='http://localhost', env="loss")

Equalize = True
## data for semi-supervised training
labeled_data = ISICdata(root=root, model='labeled', mode='semi', transform=True,
                        dataAugment=True, equalize=Equalize)
unlabeled_data = ISICdata(root=root, model='unlabeled', mode='semi', transform=True,
                          dataAugment=False, equalize=Equalize)
test_data = ISICdata(root=root, model='test', mode='semi', transform=True,
                     dataAugment=False, equalize=Equalize)

labeled_loader = DataLoader(labeled_data, batch_size=batch_size, shuffle=True,
                            num_workers=number_workers, pin_memory=True)
unlabeled_loader = DataLoader(unlabeled_data, batch_size=batch_size, shuffle=False,
                              num_workers=number_workers, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                         num_workers=number_workers, pin_memory=True)

## networks and optimisers
net = Enet(class_number)  # Enet network
unet = UNet(class_number)  # UNet network
segnet = SegNet(class_number)  # SegNet network

nets = [net, unet, segnet]

for i, net_i in enumerate(nets):
    nets[i] = net_i if (torch.cuda.is_available() and use_cuda) else net_i

if (use_cuda and torch.cuda.is_available()):
    for i, net_i in enumerate(nets):
        nets[net_i] = torch.nn.DataParallel(net_i)
    cudnn.benchmark = True
map_location = lambda storage, loc: storage

optiENet = torch.optim.Adam(nets[0].parameters(), lr=lr, weight_decay=weigth_decay)

# verify if UNet and SegNet are pre-trained models
optiUNet = torch.optim.Adam(nets[1].parameters(), lr=lr, weight_decay=weigth_decay)
optiSegNet = torch.optim.Adam(nets[2].parameters(), lr=lr, weight_decay=weigth_decay)

optimizers = [optiENet, optiUNet, optiSegNet]

## loss
class_weigth = [1 * 0.1, 3.53]
class_weigth = torch.Tensor(class_weigth)
criterion = CrossEntropyLoss2d(class_weigth).cuda() if (torch.cuda.is_available() and use_cuda) else CrossEntropyLoss2d(
    class_weigth)
ensemble_criterion = JensenShannonDivergence(reduce=True, size_average=False)
highest_iou_enet = -1
highest_iou_unet = -1
highest_iou_segnet = -1


def pre_train():
    """
    This function performs the training with the unlabeled images.
    """
    for net_i in nets:
        net_i.train()

    iou_meters = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]
    loss_meters = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]
    for epoch in range(max_epoch):

        for idx, _ in enumerate(nets):
            iou_meters[idx].reset()
            loss_meters[idx].reset()
        if epoch % 5 == 0:
            for opti_i in optimizers:
                for param_group in opti_i.param_groups:
                    param_group['lr'] = param_group['lr'] * (0.95 ** (epoch // 10))
                    print('learning rate:', param_group['lr'])

        for i, (img, mask, _) in tqdm(enumerate(labeled_loader)):
            (img, mask) = (img.cuda(), mask.cuda()) if (torch.cuda.is_available() and use_cuda) else (img, mask)

            for idx, net_i in enumerate(nets):
                optimizers[idx].zero_grad()
                pred = nets[idx](img)
                loss = criterion(pred, mask.squeeze(1))
                loss.backward()
                optimizers[idx].step()
                loss_meters[idx].add(loss.item())
                iou = iou_loss(pred2segmentation(pred), mask.squeeze(1).float(), class_number)[1]
                loss_meters[idx].add(loss.item())
                iou_meters[idx].add(iou)

                if i % train_print_frequncy == 0:
                    showImages(board_train_image, img, mask, pred2segmentation(pred))
            for idx, _ in enumerate(nets):
                if idx == 0:
                    board_loss.plot('train_iou_per_epoch for ENet', iou_meters[idx].value()[0])
                    board_loss.plot('train_loss_per_epoch ENet', loss_meters[idx].value()[0])
                elif idx == 1:
                    board_loss.plot('train_iou_per_epoch UNet', iou_meters[idx].value()[0])
                    board_loss.plot('train_loss_per_epoch UNet', loss_meters[idx].value()[0])
                else:
                    board_loss.plot('train_iou_per_epoch SegNet', iou_meters[idx].value()[0])
                    board_loss.plot('train_loss_per_epoch SegNet', loss_meters[idx].value()[0])

            for idx, net_i in enumerate(nets):
                if idx == 0:
                    torch.save(net_i.state_dict(), 'checkpoint/modified_ENet_pre-trained.pth')
                elif idx == 1:
                    torch.save(net_i.state_dict(), 'checkpoint/modified_UNet_pre-trained.pth')
                else:
                    torch.save(net_i.state_dict(), 'checkpoint/modified_SegNet_pre-trained.pth')

            train(nets, labeled_loader, unlabeled_loader)


def train(nets_, labeled_loader_, unlabeled_loader_):
    """
    This function performs the training with the unlabeled subset.
    """
    for net_i in nets_:
        net_i.train()

    labeled_loader_iter = enumerate(labeled_loader_)
    unlabeled_loader_iter = enumerate(unlabeled_loader_)

    iou_meters = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]
    loss_meters = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]
    loss_ensemble_meter = AverageValueMeter()

    for epoch in range(max_epoch):

        for idx, _ in enumerate(nets_):
            iou_meters[idx].reset()
            loss_meters[idx].reset()
        if epoch % 5 == 0:
            for opti_i in optimizers:
                for param_group in opti_i.param_groups:
                    param_group['lr'] = param_group['lr'] * (0.95 ** (epoch // 10))
                    print('learning rate:', param_group['lr'])

        # train with labeled data
        try:
            _, labeled_batch = labeled_loader_iter.__next__()
        except:
            labeled_loader_iter = enumerate(labeled_loader_)
            _, labeled_batch = labeled_loader_iter.__next__()

        img, mask, _ = labeled_batch
        (img, mask) = (img.cuda(), mask.cuda()) if (torch.cuda.is_available() and use_cuda) else (img, mask)

        optiENet.zero_grad()
        optiUNet.zero_grad()
        optiSegNet.zero_grad()

        pred_enet = nets[0](img)
        pred_unet = nets[1](img)
        pred_segnet = nets[2](img)

        loss_enet = criterion(pred_enet, mask.squeeze(1))
        loss_unet = criterion(pred_unet, mask.squeeze(1))
        loss_segnet = criterion(pred_segnet, mask.squeeze(1))

        loss_enet.backward()
        loss_unet.backward()
        loss_segnet.backward()

        optiENet.step()
        optiUNet.step()
        optiSegNet.step()

        # computing loss
        loss_ensemble_meter.add(loss_enet.item())
        loss_ensemble_meter.add(loss_unet.item())
        loss_ensemble_meter.add(loss_segnet.item())

        # train with unlabeled data
        try:
            _, unlabeled_batch = unlabeled_loader_iter.__next__()
        except:
            unlabeled_loader_iter = enumerate(unlabeled_loader_)
            _, unlabeled_batch = unlabeled_loader_iter.__next__()

        img, _, _ = unlabeled_batch
        img = img.cuda() if (torch.cuda.is_available() and use_cuda) else img

        optimizers[0].zero_grad()  # ENet optimizer
        optimizers[1].zero_grad()  # UNet optimizer
        optimizers[2].zero_grad()  # SegNet optimizer
        # computing nets output
        enet_prob = F.softmax(nets_[0](img), dim=1)  # ENet output
        unet_prob = F.softmax(nets_[1](img), dim=1)  # UNet output
        segnet_prob = F.softmax(nets_[1](img), dim=1)  # SegNet output
        ensemble_probs = torch.cat([enet_prob, unet_prob, segnet_prob], 0)

        jsd_loss = ensemble_criterion(ensemble_probs)
        print(jsd_loss)

        loss_ensemble_meter.add(jsd_loss.item())

        board_loss.plot('train_loss_per_epoch SegNet', loss_ensemble_meter.value()[0])

        # testing segmentation nets
        test(nets, test_loader)


def test(nets_, test_loader_):
    """
    This function performs the evaluation with the test set containing labeled images.
    """
    global highest_iou
    for net_i in nets_:
        net_i.eval()
    iou_meters_test = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]
    loss_meters_test = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]
    for idx, _ in enumerate(nets_):
        iou_meters_test[idx].reset()
    for i, (img, mask, _) in tqdm(enumerate(test_loader_)):
        (img, mask) = (img.cuda(), mask.cuda()) if (torch.cuda.is_available() and use_cuda) else (img, mask)

        for idx, net_i in enumerate(nets):
            optimizers[idx].zero_grad()
            pred_test = nets[idx](img)
            loss_test = criterion(pred_test, mask.squeeze(1))
            loss_test.backward()
            optimizers[idx].step()
            loss_meters_test[idx].add(loss_test.item())
            iou_test = iou_loss(pred2segmentation(pred_test), mask.squeeze(1).float(), class_number)[1]
            #loss_meters_test[idx].add(loss.item())
            iou_meters_test[idx].add(iou_test)
        if i % val_print_frequncy == 0:
            showImages(board_test_image, img, mask, pred2segmentation(pred_test))

    for idx, _ in enumerate(nets):
        if idx == 0:
            board_loss.plot('test_iou_per_epoch for ENet', iou_meters_test[idx].value()[0])
            board_loss.plot('test_loss_per_epoch for ENet', loss_meters_test[idx].value()[0])
        elif idx == 1:
            board_loss.plot('test_iou_per_epoch for UNet', iou_meters_test[idx].value()[0])
            board_loss.plot('test_loss_per_epoch for UNet', loss_meters_test[idx].value()[0])
        else:
            board_loss.plot('test_iou_per_epoch for SegNet', iou_meters_test[idx].value()[0])
            board_loss.plot('test_loss_per_epoch for SegNet', loss_meters_test[idx].value()[0])

    net.train()
    for idx, _ in enumerate(nets):
        if idx == 0:
            if highest_iou_enet < iou_meters_test[idx].value()[0]:
                highest_iou = iou_meters_test[idx].value()[0]
                torch.save(nets[idx].state_dict(),
                           'checkpoint/modified_ENet_%.3f_%s.pth' % (
                           iou_meters_test[idx].value()[0], 'equal_' + str(Equalize)))
                print('The highest IOU is:%.3f' % iou_meters_test[idx].value()[0], 'Model saved.')
        elif idx == 1:
            if highest_iou_unet < iou_meters_test[idx].value()[0]:
                highest_iou = iou_meters_test[idx].value()[0]
                torch.save(nets[idx].state_dict(),
                           'checkpoint/modified_ENet_%.3f_%s.pth' % (
                           iou_meters_test[idx].value()[0], 'equal_' + str(Equalize)))
                print('The highest IOU is:%.3f' % iou_meters_test[idx].value()[0], 'Model saved.')
        else:
            if highest_iou_segnet < iou_meters_test[idx].value()[0]:
                highest_iou = iou_meters_test[idx].value()[0]
                torch.save(nets[idx].state_dict(),
                           'checkpoint/modified_ENet_%.3f_%s.pth' % (
                           iou_meters_test[idx].value()[0], 'equal_' + str(Equalize)))
                print('The highest IOU is:%.3f' % iou_meters_test[idx].value()[0], 'Model saved.')


if __name__ == "__main__":
    pre_train()

