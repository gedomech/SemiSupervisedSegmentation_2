import os
import sys
import pandas as pd
import logging
from tensorboardX import SummaryWriter

from myutils.myDataLoader import ISICdata
from myutils.myENet import Enet

from myutils.myLoss import CrossEntropyLoss2d
import warnings
from myutils.myUtils import *


logger = logging.getLogger(__name__)
logger.parent = None
warnings.filterwarnings('ignore')

class_number = 2
lr = 1e-5
weigth_decay = 1e-6
lamda = 5e-2


use_cuda = True
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
number_workers = 0
labeled_batch_size = 2
unlabeled_batch_size = 2
val_batch_size = 1

max_epoch_ensemble = 100

class_weigth = [1, 1]
class_weigth = torch.Tensor(class_weigth)

## networks and optimisers
nets = [Enet(class_number),
        Enet(class_number)]

nets = map_(lambda x: x.to(device), nets)

map_location = lambda storage, loc: storage

optimizers = [torch.optim.Adam(nets[0].parameters(), lr=lr, weight_decay=weigth_decay),
              torch.optim.Adam(nets[1].parameters(), lr=lr, weight_decay=weigth_decay)]

criterion = CrossEntropyLoss2d(class_weigth).to(device) if (
        torch.cuda.is_available() and use_cuda) else CrossEntropyLoss2d(
    class_weigth)
ensemble_criterion = JensenShannonDivergence(reduce=True, size_average=False)

historical_score_dict = {
    'epoch': -1,
    'enet_0': 0,
    'enet_2': 0,
    'mv': 0,
    'jsd': 0}


def get_loss(predictions):
    p = torch.cat(predictions)
    criteron = ensemble_criterion
    loss = criteron(p)
    return loss


def get_exclusive_dataloaders(lab_percent=0.04, num_workers=4, batch_size=4, shuffle=True):
    root_path = "datasets/ISIC2018"
    labeled_data = ISICdata(root=root_path, model='labeled', mode='semi', transform=True,
                            dataAugment=False, equalize=False)
    labeled_data1 = ISICdata(root=root_path, model='labeled', mode='semi', transform=True,
                             dataAugment=False, equalize=False)
    labeled_data2 = ISICdata(root=root_path, model='labeled', mode='semi', transform=True,
                             dataAugment=False, equalize=False)

    n_imgs_per_set = int(len(labeled_data.imgs) * lab_percent)

    if shuffle:
        # randomizing the list of images and gts
        np.random.seed(1)
        data = list(zip(labeled_data.imgs, labeled_data.gts))
        np.random.shuffle(data)
        labeled_data.imgs, labeled_data.gts = zip(*data)

    # creating the 3 exclusive labeled sets
    labeled_data1.imgs = labeled_data.imgs[:n_imgs_per_set]
    labeled_data1.gts = labeled_data.gts[:n_imgs_per_set]
    labeled_data2.imgs = labeled_data.imgs[n_imgs_per_set:2 * n_imgs_per_set]
    labeled_data2.gts = labeled_data.gts[n_imgs_per_set:2 * n_imgs_per_set]
    labeled_data.imgs = labeled_data.imgs[2 * n_imgs_per_set:3 * n_imgs_per_set]
    labeled_data.gts = labeled_data.gts[2 * n_imgs_per_set:3 * n_imgs_per_set]

    unlabeled_data = ISICdata(root=root_path, model='unlabeled', mode='semi', transform=True,
                              dataAugment=False, equalize=False)
    val_data = ISICdata(root=root_path, model='val', mode='semi', transform=True,
                        dataAugment=False, equalize=False)

    labeled_loader_params = {'batch_size': num_workers,
                             'shuffle': True,  # True
                             'num_workers': batch_size,
                             'pin_memory': True}

    unlabeled_loader_params = {'batch_size': num_workers,
                               'shuffle': True,  # True
                               'num_workers': batch_size,
                               'pin_memory': True}
    val_loader_params = {'batch_size': num_workers,
                         'shuffle': False,  # True
                         'num_workers': batch_size,
                         'pin_memory': True}

    labeled_data1 = DataLoader(labeled_data1, **labeled_loader_params)
    labeled_data2 = DataLoader(labeled_data2, **labeled_loader_params)
    labeled_data = DataLoader(labeled_data, **labeled_loader_params)

    unlabeled_data = DataLoader(unlabeled_data, **unlabeled_loader_params)
    val_data = DataLoader(val_data, **val_loader_params)

    logger.info('the size of labeled_data:{}, unlabeled_data:{}, val_data:{}'.format(labeled_data.__len__(),
                                                                                     unlabeled_data.__len__(),
                                                                                     val_data.__len__()))

    return {'labeled': [labeled_data1, labeled_data2, labeled_data],
            'unlabeled': unlabeled_data,
            'val': val_data}


def train_ensemble(nets_: list, nets_path_, labeled_loader_: list, unlabeled_loader_, test_data):
    records =[]
    """
    This function performs the training of the pre-trained models with the labeled and unlabeled data.
    """
    #  loading pre-trained models

    map_(lambda x, y: [x.load_state_dict(torch.load(y, map_location='cpu')), x.train()], nets_, nets_path_)
    global historical_score_dict
    nets_path = ['checkpoint/best_enet_0.pth',
                 'checkpoint/best_enet_2.pth']
    dice_meters = [AverageValueMeter(), AverageValueMeter()]

    print("STARTING THE BASELINE TRAINING!!!!")
    for epoch in range(max_epoch_ensemble):
        print('epoch = {0:4d}/{1:4d} training baseline'.format(epoch, max_epoch_ensemble))

        # if epoch % 5 == 0:
        #     learning_rate_decay(optimizers, 0.95)

        # train with labeled data
        for _ in range(len(unlabeled_loader_)):

            imgs, masks, _ = image_batch_generator(labeled_loader_, device=device)
            _, llost_list, dice_score = batch_labeled_loss_(imgs, masks, nets_, criterion)

            # train with unlabeled data
            imgs, _, _ = image_batch_generator(unlabeled_loader_, device=device)
            pseudolabel, predictions = get_mv_based_labels(imgs, nets_)
            jsd_loss = get_loss(predictions)

            total_loss = [x + jsd_loss for x in llost_list]
            for idx in range(len(optimizers)):
                optimizers[idx].zero_grad()
                total_loss[idx].backward()
                optimizers[idx].step()
                dice_meters[idx].add(dice_score[idx])

        print(
            'train epoch {0:1d}/{1:d} ensemble: enet0_dice_score={2:.3f}, enet2_dice_score={3:.3f}'.format(
                epoch + 1, max_epoch_ensemble, dice_meters[0].value()[0], dice_meters[1].value()[0]))

        score_meters, ensemble_score = test(nets_, test_data, device=device)

        print(
            'val epoch {0:d}/{1:d} ensemble: enet0_dice={2:.3f}, enet2_dice={3:.3f}, with mv_dice={4:.3f}'.format(
                epoch + 1,
                max_epoch_ensemble,
                score_meters[0].value()[0],
                score_meters[1].value()[0],
                ensemble_score.value()[0]))

        historical_score_dict = save_models(nets_, nets_path, score_meters, epoch, historical_score_dict)
        if ensemble_score.value()[0] > historical_score_dict['mv']:
            historical_score_dict['mv'] = ensemble_score.value()[0]

        records.append(historical_score_dict)

        try:
            pd.DataFrame(records).to_csv('ensemble_records.csv')
        except Exception as e:
            print(e)


if __name__ == "__main__":

    nets_path = ['checkpoint/best_enet_0_semi.pth',
                 'checkpoint/best_enet_0_semi.pth']

    data_loaders = get_exclusive_dataloaders(lab_percent=0.04, num_workers=4,
                                             batch_size=4, shuffle=True)

    saved_model1 = torch.load('checkpoint/best_enet_0.pt', map_location=lambda storage, loc: storage)
    saved_model2 = torch.load('checkpoint/best_enet_1.pt', map_location=lambda storage, loc: storage)

    # networks and optimisers
    net1 = Enet(2)
    net2 = Enet(2)
    nets = [net1.load_dict(saved_model1['model']),
            net2.load_dict(saved_model2['model'])]
    _ = [x.to(device) for x in nets]

    labeled_loader = [data_loaders['labeled'][0], data_loaders['labeled'][2]]
    unlabeled_loader = data_loaders['unlabeled']
    val_loader = data_loaders['val']

    train_ensemble(nets, nets_path, labeled_loader, unlabeled_loader, val_loader)
