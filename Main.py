import sys, os

sys.path.extend([os.path.dirname(os.getcwd())])
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from myutils.myDataLoader import ISICdata
from myutils.myENet import Enet
from myutils.myNetworks import UNet, SegNet
from myutils.myLoss import CrossEntropyLoss2d, JensenShannonDivergence
import warnings
from tqdm import tqdm
from torchnet.meter import AverageValueMeter
from myutils.myUtils import pred2segmentation, iou_loss, showImages, dice_loss
from myutils.myVisualize import Dashboard
warnings.filterwarnings('ignore')

torch.set_num_threads(3)  # set by deafault to 1
root = "datasets/ISIC2018"

class_number = 2
lr = 1e-4
weigth_decay = 1e-6
use_cuda = True
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
number_workers = 0
batch_size = 1
max_epoch_pre = 0
max_epoch_baseline = 100
train_print_frequncy = 10
val_print_frequncy = 10

## visualization
# board_train_image = Dashboard(server='http://localhost', env="image_train")
# board_test_image = Dashboard(server='http://localhost', env="image_test")
# board_loss = Dashboard(server='http://localhost', env="loss")

Equalize = False
## data for semi-supervised training
labeled_data = ISICdata(root=root, model='labeled', mode='semi', transform=True,
                        dataAugment=True, equalize=Equalize)
unlabeled_data = ISICdata(root=root, model='unlabeled', mode='semi', transform=True,
                          dataAugment=False, equalize=Equalize)
test_data = ISICdata(root=root, model='test', mode='semi', transform=True,
                     dataAugment=False, equalize=Equalize)

labeled_data = DataLoader(labeled_data, batch_size=batch_size, shuffle=True,
                            num_workers=number_workers, pin_memory=True)
unlabeled_data = DataLoader(unlabeled_data, batch_size=batch_size, shuffle=False,
                              num_workers=number_workers, pin_memory=True)
test_data = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                         num_workers=number_workers, pin_memory=True)

## networks and optimisers
net = Enet(class_number)  # Enet network
unet = UNet(class_number)  # UNet network
segnet = SegNet(class_number)  # SegNet network

nets = [net, unet, segnet]

for i, net_i in enumerate(nets):
    nets[i] = net_i.to(device) if (torch.cuda.is_available() and use_cuda) else net_i

map_location = lambda storage, loc: storage

optiENet = torch.optim.Adam(nets[0].parameters(), lr=lr, weight_decay=weigth_decay)
optiUNet = torch.optim.Adam(nets[1].parameters(), lr=lr, weight_decay=weigth_decay)
optiSegNet = torch.optim.Adam(nets[2].parameters(), lr=lr, weight_decay=weigth_decay)
optimizers = [optiENet, optiUNet, optiSegNet]

## loss
class_weigth = [1 * 0.1, 3.53]
class_weigth = torch.Tensor(class_weigth)
criterion = CrossEntropyLoss2d(class_weigth).to(device) if (
            torch.cuda.is_available() and use_cuda) else CrossEntropyLoss2d(
    class_weigth)
ensemble_criterion = JensenShannonDivergence(reduce=True, size_average=False)

highest_dice_enet = -1
highest_dice_unet = -1
highest_dice_segnet = -1
highest_mv_dice_score = -1

def batch_iteration(dataloader:DataLoader)->tuple:
    try:
        _, labeled_batch = enumerate(dataloader).__next__()
    except:
        labeled_loader_iter = enumerate(dataloader)
        _, labeled_batch = labeled_loader_iter.__next__()
    return labeled_batch



def pre_train():
    """
    This function performs the training with the unlabeled images.
    """
    for net_i in nets:
        net_i.train()

    global highest_dice_enet
    global highest_dice_unet
    global highest_dice_segnet
    nets_path = ['checkpoint/best_ENet_pre-trained.pth',
                 'checkpoint/best_UNet_pre-trained.pth',
                 'checkpoint/best_SegNet_pre-trained.pth']
    dice_meters = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]

    for epoch in range(max_epoch_pre):
        print('epoch = {0:4d}/{1:4d} pre-training'.format(epoch, max_epoch_pre))
        for idx, _ in enumerate(nets):
            dice_meters[idx].reset()


        if epoch % 5 == 0:
            for opti_i in optimizers:
                for param_group in opti_i.param_groups:
                    param_group['lr'] = param_group['lr'] * (0.95)
                    print('learning rate:', param_group['lr'])

        for i, (img, mask, _) in tqdm(enumerate(labeled_data)):
            img, mask = img.to(device), mask.to(device)

            for idx, net_i in enumerate(nets):
                optimizers[idx].zero_grad()
                pred = nets[idx](img)
                loss = criterion(pred, mask.squeeze(1))
                loss.backward()
                optimizers[idx].step()
        test(nets, nets_path, test_data)

    train_baseline(nets, nets_path, labeled_data, unlabeled_data)


def train_baseline(nets_, nets_path_, labeled_loader_:DataLoader, unlabeled_loader_:DataLoader, method='A'):
    """
    This function performs the training of the pre-trained models with the labeled and unlabeled data.
    """
    # loading pre-trained models
    for idx, net_i in enumerate(nets_):
        net_i.load_state_dict(torch.load(nets_path_[idx]))
        net_i.train()

    global highest_mv_dice_score
    nets_path = ['checkpoint/best_ENet_baseline.pth',
                 'checkpoint/best_UNet_baseline.pth',
                 'checkpoint/best_SegNet_baseline.pth']

    dice_meters = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]
    loss_meters = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]
    for epoch in range(max_epoch_baseline):
        print('epoch = {0:4d}/{1:4d} training baseline'.format(epoch, max_epoch_baseline))
        for idx, _ in enumerate(nets_):
            dice_meters[idx].reset()
            loss_meters[idx].reset()
        if epoch % 5 == 0:
            for opti_i in optimizers:
                for param_group in opti_i.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.95
                    print('learning rate:', param_group['lr'])

        # train with labeled data
        labeled_batch = batch_iteration(labeled_loader_)

        img, mask, _ = labeled_batch
        img, mask = img.to(device), mask.to(device)
        lloss_list= []

        for idx, net_i in enumerate(nets):

            optimizers[idx].zero_grad()
            pred = nets[idx](img)
            labeled_loss = criterion(pred, mask.squeeze(1))
            dice_score = dice_loss(pred2segmentation(pred), mask.squeeze(1))
            dice_meters[idx].add(dice_score)
            if method != 'A':
                labeled_loss.backward()
                optimizers[idx].step()
            if method=='A':
                lloss_list.append(labeled_loss)


        # train with unlabeled data
        unlabeled_batch = batch_iteration(unlabeled_loader_)

        img, _, _ = unlabeled_batch
        img = img.to(device)
        # computing the majority voting from the output nets
        distributions = torch.zeros([img.shape[0], class_number, img.shape[2], img.shape[3]])
        for idx, net_i in enumerate(nets):
            pred = nets[idx](img)
            distributions += F.softmax(pred.cpu(), 1)

        distributions /= 3
        u_loss = []

        for idx, net_i in enumerate(nets):

            pred = nets[idx](img)
            unlabled_loss = criterion(pred, pred2segmentation(distributions.to(device)))
            if method!='A':
                optimizers[idx].zero_grad()
                unlabled_loss.backward()
                optimizers[idx].step()
            elif method=='A':
                u_loss.append(unlabled_loss)

        if method =='A':
            for idx in range(3):
                optimizers[idx].zero_grad()
                total_loss = lloss_list[idx] + u_loss[idx]
                total_loss.backward()
                optimizers[idx].step()






        test(nets_, nets_path, test_data, method='A')


def test(nets_, nets_path_, test_loader_):
    """
    This function performs the evaluation with the test set containing labeled images.
    """
    global highest_dice_enet
    global highest_dice_unet
    global highest_dice_segnet

    for i, net_i in enumerate(nets_):
        nets_[i].eval()

    dice_meters_test = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]
    mv_dice_score_meter = AverageValueMeter()

    for idx, _ in enumerate(nets_):
        dice_meters_test[idx].reset()
    for i, (img, mask, _) in tqdm(enumerate(test_loader_)):
        (img, mask) = img.to(device), mask.to(device)
        distributions = torch.zeros([img.shape[0], class_number, img.shape[2], img.shape[3]])
        for idx, net_i in enumerate(nets):
            pred_test = nets[idx](img)
            distributions += F.softmax(pred_test, 1)
            dice_test = dice_loss(pred2segmentation(pred_test), mask.squeeze(1))
            dice_meters_test[idx].add(dice_test)

        distributions /= 3
        mv_dice_score = dice_loss(pred2segmentation(distributions.to(device)), mask.squeeze(1))
        mv_dice_score_meter.add(mv_dice_score.item())
        # if i % val_print_frequncy == 0:
        #     showImages(board_test_image, img, mask, pred2segmentation(distributions))

    for net_i in nets_:
        net_i.train()

    for idx, net_i in enumerate(nets_):

        if (idx == 0) and (highest_dice_enet < mv_dice_score_meter.value()[0]):
            highest_dice_enet = mv_dice_score_meter.value()[0]
            print('The highest dice for ENet is {:.3f}'.format(highest_dice_enet))
            torch.save(net_i.state_dict(), nets_path_[idx])

        elif (idx == 1) and (highest_dice_unet < mv_dice_score_meter.value()[0]):
            highest_dice_unet = mv_dice_score_meter.value()[0]
            print('The highest dice for UNet is {:.3f}'.format(highest_dice_unet))
            torch.save(net_i.state_dict(), nets_path_[idx])

        elif (idx == 2) and (highest_dice_segnet < mv_dice_score_meter.value()[0]):
            highest_dice_segnet = mv_dice_score_meter.value()[0]
            print('The highest dice for SegNet is {:.3f}'.format(highest_dice_segnet))
            torch.save(net_i.state_dict(), nets_path_[idx])


if __name__ == "__main__":
    pre_train()
    # # nets_path = ['checkpoint/best_ENet_pre-trained.pth',
    #              'checkpoint/best_UNet_pre-trained.pth',
    #              'checkpoint/best_SegNet_pre-trained.pth']
    # train_baseline(nets, nets_path, labeled_loader, unlabeled_loader)
