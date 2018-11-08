# coding=utf-8
import logging
import os
import sys

import pandas as pd

logging.basicConfig(format='%(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger('spam_application')
logger.setLevel(logging.INFO)
sys.path.extend([os.path.dirname(os.getcwd())])
from myutils.myDataLoader import ISICdata
from myutils.myENet import Enet
from myutils.myLoss import CrossEntropyLoss2d
import warnings
from tqdm import tqdm
from myutils.myUtils import *

warnings.filterwarnings('ignore')
# writer = SummaryWriter()

# torch.set_num_threads(1)  # set by deafault to 1
root = "../datasets/ISIC2018"

class_number = 2
lr = 1e-3
weigth_decay = 1e-5
lamda = 1

use_cuda = True
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
number_workers = 4
labeled_batch_size = 4
unlabeled_batch_size = 4
val_batch_size = 4

max_epoch_pre = 200
max_epoch_baseline = 200
max_epoch_ensemble = 100
train_print_frequncy = 10
val_print_frequncy = 10

Equalize = False
## data for semi-supervised training
labeled_data = ISICdata(root=root, model='labeled', mode='semi', transform=True,
                        dataAugment=False, equalize=Equalize)
unlabeled_data = ISICdata(root=root, model='unlabeled', mode='semi', transform=True,
                          dataAugment=False, equalize=Equalize)
dev_data = ISICdata(root=root, model='dev', mode='semi', transform=True,
                    dataAugment=False, equalize=Equalize)
val_data = ISICdata(root=root, model='val', mode='semi', transform=True,
                    dataAugment=False, equalize=Equalize)

labeled_loader_params = {'batch_size': labeled_batch_size,
                         'shuffle': True,
                         'num_workers': number_workers,
                         'pin_memory': True}

unlabeled_loader_params = {'batch_size': unlabeled_batch_size,
                           'shuffle': False,
                           'num_workers': number_workers,
                           'pin_memory': True}

labeled_data = DataLoader(labeled_data, **labeled_loader_params)
unlabeled_data = DataLoader(unlabeled_data, **unlabeled_loader_params)
dev_data = DataLoader(dev_data, **unlabeled_loader_params)
val_data = DataLoader(val_data, **unlabeled_loader_params)
map_location=lambda storage, loc: storage
## networks and optimisers
net = Enet(class_number)
net = net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weigth_decay)

## loss
class_weigth = [1 * 0.1, 3.53]
class_weigth = torch.Tensor(class_weigth)
criterion = CrossEntropyLoss2d(class_weigth).to(device)
ensemble_criterion = JensenShannonDivergence(reduce=True, size_average=False)
historical_track = []


def pre_train(p):
    net_save_path = 'enet_pretrained_%.1f.pth' % float(p)

    labeled_len = int(labeled_data.dataset.imgs.__len__() * float(p))
    labeled_data.dataset.imgs = labeled_data.dataset.imgs[:labeled_len]
    labeled_data.dataset.gts = labeled_data.dataset.gts[:labeled_len]
    print('the length of the labeled dataset is: %d' % labeled_len)
    best_dev_score = -1
    schduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 175], gamma=0.2)

    for epoch in range(max_epoch_pre):
        schduler.step()
        for i, (img, mask, _) in tqdm(enumerate(labeled_data)):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = net(img)
            loss = criterion(pred, mask.squeeze(1))
            loss.backward()
            optimizer.step()

        [labeled_score, unlabeled_score, dev_score, validation_score] = [evaluate(net, x, device) for x in
                                                                         (labeled_data, unlabeled_data, dev_data,
                                                                          val_data)]

        logging.info('pretrained stage: lab:%3f  unlab:%3f  dev:%3f   val:%.3f' % (
            labeled_score, unlabeled_score, dev_score, validation_score))
        historical_track.append(
            {'lab': labeled_score, 'unlab': unlabeled_score, 'val': validation_score, 'dev': dev_score, 'epoch': epoch})
        pd.DataFrame(historical_track).to_csv(net_save_path.replace('pth', 'csv'))

        if dev_score > best_dev_score:
            dict_to_save = {'labeled_dataloader': labeled_data,
                            'state_dict': net.state_dict()}
            torch.save(dict_to_save, net_save_path)
            best_dev_score = dev_score
    return net_save_path, best_dev_score


def train_baseline(net_, net_path_,separete_backpr ='False'):
    assert separete_backpr in ('True', 'False')
    semi_historical_track = []
    """
    This function performs the training of the pre-trained models with the labeled and unlabeled data.
    """
    #  loading pre-trained models
    net_.load_state_dict(torch.load(net_path_, map_location= map_location)['state_dict'])
    labeled_data = torch.load(net_path_,map_location=map_location )['labeled_dataloader']
    print('the length of the labeled dataset is: %d' % labeled_data.dataset.imgs.__len__())
    net_.train()
    learning_rate_reset(optimizer, lr=1e-6)

    best_dev_score = -1
    print("STARTING THE BASELINE TRAINING!!!!")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 130, 160, 180], gamma=0.5)
    for epoch in range(max_epoch_baseline):
        print('epoch = {0:4d}/{1:4d} training baseline'.format(epoch, max_epoch_baseline))
        scheduler.step()

        [labeled_score, unlabeled_score, dev_score, validation_score] = [evaluate(net, x, device) for x in
                                                                         (labeled_data, unlabeled_data, dev_data,
                                                                          val_data)]
        semi_historical_track.append(
            {'lab': labeled_score, 'unlab': unlabeled_score, 'val': validation_score, 'dev': dev_score, 'epoch': epoch})
        # train with labeled data
        for _ in tqdm(range(len(labeled_data))):  #

            imgs, masks, _ = image_batch_generator(labeled_data, device=device)
            _, llost_list, _ = batch_labeled_loss_(imgs, masks, [net_], criterion)
            if separete_backpr=='True':
                optimizer.zero_grad()
                llost_list[0].backward()
                optimizer.step()

            # train with unlabeled data
            imgs, _, _ = image_batch_generator(unlabeled_data, device=device)
            pseudolabel, predictions = get_mv_based_labels(imgs, [net_])
            ulost_list = cotraining(predictions, pseudolabel, [net_], criterion, device)
            if separete_backpr=='True':
                optimizer.zero_grad()
                ulost_list[0].backward()
                optimizer.step()
            if not separete_backpr=='True':
                total_loss = [x + lamda * y for x, y in zip(llost_list, ulost_list)]
                optimizer.zero_grad()
                total_loss[0].backward()
                optimizer.step()

        pd.DataFrame(semi_historical_track).to_csv(net_path_.replace('pretrained', 'baseline').replace('pth', 'csv').split('.csv')[0]+'_separate_'+str(separete_backpr)+'.csv')

        if dev_score > best_dev_score:
            torch.save(net.state_dict(), net_path_.replace('pretrained', 'baseline'))
            best_dev_score = dev_score


if __name__ == "__main__":
    # Pre-training Stage
    import argparse

    parser = argparse.ArgumentParser(description='split the training data')
    parser.add_argument('--p', default=0.5)
    parser.add_argument('--pretrain', default='False')
    parser.add_argument('--baseline', default='True')
    parser.add_argument('--separate_backpro', default='False')
    args = parser.parse_args()
    if args.pretrain=='True':
        saved_path, pretrained_score = pre_train(args.p)
        if args.baseline==True:
            train_baseline(net, saved_path)
    elif args.baseline=='True':
        saved_path = 'enet_pretrained_%.1f.pth' % float(args.p)
        train_baseline(net, saved_path, args.separate_backpro)
