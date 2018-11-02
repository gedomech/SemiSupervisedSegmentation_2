# coding=utf-8
import logging
import os
import sys

import pandas as pd

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
lr = 1e-4
weigth_decay = 1e-6
lamda = 5e-2

use_cuda = True
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
number_workers = 2
labeled_batch_size = 2
unlabeled_batch_size = 2
val_batch_size = 1

max_epoch_pre = 50
max_epoch_baseline = 50
max_epoch_ensemble = 100
train_print_frequncy = 10
val_print_frequncy = 10

Equalize = False
## data for semi-supervised training
labeled_data = ISICdata(root=root, model='labeled', mode='semi', transform=True,
                        dataAugment=False, equalize=Equalize)
unlabeled_data = ISICdata(root=root, model='unlabeled', mode='semi', transform=True,
                          dataAugment=False, equalize=Equalize)
test_data = ISICdata(root=root, model='test', mode='semi', transform=True,
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
test_data = DataLoader(test_data, **unlabeled_loader_params)

## networks and optimisers
net = Enet(class_number)
net = net.to(device)

map_location = lambda storage, loc: storage
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weigth_decay)

## loss
class_weigth = [1 * 0.1, 3.53]
class_weigth = torch.Tensor(class_weigth)
criterion = CrossEntropyLoss2d(class_weigth).to(device) if (
        torch.cuda.is_available() and use_cuda) else CrossEntropyLoss2d(
    class_weigth)
ensemble_criterion = JensenShannonDivergence(reduce=True, size_average=False)
historical_track= []


def pre_train(p):

    net_save_path = 'enet_pretrained_%.1f.pth'%float(p)

    labeled_len = int(labeled_data.dataset.imgs.__len__()*float(p))
    labeled_data.dataset.imgs = labeled_data.dataset.imgs[:labeled_len]
    labeled_data.dataset.gts = labeled_data.dataset.gts[:labeled_len]
    print('the length of the labeled dataset is: %d'%labeled_len)
    best_validation_score = -1

    for epoch in range(max_epoch_pre):

        if epoch +1 % 5 == 0:
            learning_rate_decay(optimizer, 0.95)

        for i, (img, mask, _) in tqdm(enumerate(labeled_data)):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = net(img)
            loss = criterion(pred, mask.squeeze(1))
            loss.backward()
            optimizer.step()

        [labeled_score,unlabeled_score, validation_score] = [evaluate(net, x) for x in (labeled_data,unlabeled_data,test_data)]

        logging.info('pretrained stage: lab:%3f  unlab:%3f  val:%.3f'%(labeled_score,unlabeled_score,validation_score))
        historical_track.append({'lab':labeled_score,'unlab':unlabeled_score,'val':validation_score,'epoch':epoch})
        pd.DataFrame(historical_track).to_csv(net_save_path.replace('pth','csv'))

        if validation_score> best_validation_score:
            dict_to_save = {'labeled_dataloader': labeled_data,
                            'state_dict':net.state_dict()}
            torch.save(dict_to_save,net_save_path)
            best_validation_score = validation_score
    return net_save_path, best_validation_score



def train_baseline(net_, net_path_):
    semi_historical_track = []
    """
    This function performs the training of the pre-trained models with the labeled and unlabeled data.
    """
    #  loading pre-trained models
    net_.load_state_dict(torch.load(net_path_, )['state_dict'])
    labeled_data =torch.load(net_path_, )['labeled_dataloader']
    print('the length of the labeled dataset is: %d'%labeled_data.dataset.imgs.__len__())
    net_.train()
    learning_rate_reset(optimizer, lr = 1e-5)

    best_validation_score = -1
    print("STARTING THE BASELINE TRAINING!!!!")
    for epoch in range(max_epoch_baseline):
        print('epoch = {0:4d}/{1:4d} training baseline'.format(epoch, max_epoch_baseline))

        if epoch+1 % 5 == 0:
            learning_rate_decay(optimizer, 0.95)

        # train with labeled data
        for _ in tqdm(range(4)):  #len(unlabeled_data)

            imgs, masks, _ = image_batch_generator(labeled_data, device=device)
            _, llost_list, _ = batch_labeled_loss_(imgs, masks, [net_], criterion)

            # train with unlabeled data
            imgs, _, _ = image_batch_generator(unlabeled_data, device=device)
            pseudolabel, predictions = get_mv_based_labels(imgs, [net_])
            ulost_list = cotraining(predictions, pseudolabel, [net_], criterion, device)
            total_loss = [x + lamda * y for x, y in zip(llost_list, ulost_list)]
            optimizer.zero_grad()
            total_loss[0].backward()
            optimizer.step()
        [labeled_score,unlabeled_score, validation_score] = [evaluate(net, x) for x in (labeled_data,unlabeled_data,test_data)]
        semi_historical_track.append(
            {'lab': labeled_score, 'unlab': unlabeled_score, 'val': validation_score, 'epoch': epoch})
        pd.DataFrame(semi_historical_track).to_csv(net_path_.replace('pretrained','baseline').replace('pth','csv'))


        if validation_score> best_validation_score:
            torch.save(net.state_dict(),net_path_.replace('pretrained','baseline'))
            best_validation_score = validation_score



if __name__ == "__main__":
    # Pre-training Stage
    import argparse
    parser = argparse.ArgumentParser(description='split the training data')
    parser.add_argument('--p',default=0.8)
    args = parser.parse_args()
    saved_path, pretrained_score = pre_train(args.p)
    train_baseline(net,saved_path)





