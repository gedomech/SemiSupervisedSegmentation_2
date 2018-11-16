# coding=utf-8
import logging
import os
import sys, json

import pandas as pd

logging.basicConfig(format='%(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger('spam_application')
logger.setLevel(logging.INFO)
sys.path.extend([os.path.dirname(os.getcwd())])
from myutils.myDataLoader import ISICdata
from myutils.myENet import Enet
from myutils.myLoss import CrossEntropyLoss2d
from tqdm import tqdm
from myutils.myUtils import *
from tensorboardX import SummaryWriter

writer = SummaryWriter()
warnings.filterwarnings('ignore')

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
max_epoch_baseline = 100
max_epoch_ensemble = 100
train_print_frequncy = 10
val_print_frequncy = 10

## data for semi-supervised training
labeled_data = ISICdata(root=root, model='labeled', mode='semi', transform=True,
                        dataAugment=False, equalize=False)
unlabeled_data = ISICdata(root=root, model='unlabeled', mode='semi', transform=True,
                          dataAugment=False, equalize=False)
dev_data = ISICdata(root=root, model='dev', mode='semi', transform=True,
                    dataAugment=False, equalize=False)
val_data = ISICdata(root=root, model='val', mode='semi', transform=True,
                    dataAugment=False, equalize=False)

labeled_loader_params = {'batch_size': labeled_batch_size,
                         'shuffle': True,  # True
                         'num_workers': number_workers,
                         'pin_memory': True}

unlabeled_loader_params = {'batch_size': unlabeled_batch_size,
                           'shuffle': True,
                           'num_workers': number_workers,
                           'pin_memory': True}

labeled_data = DataLoader(labeled_data, **labeled_loader_params)
unlabeled_data = DataLoader(unlabeled_data, **unlabeled_loader_params)
dev_data = DataLoader(dev_data, **unlabeled_loader_params)
val_data = DataLoader(val_data, **unlabeled_loader_params)
map_location = lambda storage, loc: storage
## networks and optimisers
net = Enet(class_number)
net = net.to(device)

## loss
class_weigth = [1, 1]
class_weigth = torch.Tensor(class_weigth)
criterion = CrossEntropyLoss2d(class_weigth).to(device)
ensemble_criterion = JensenShannonDivergence(reduce=True, size_average=False)
historical_track = []


def pre_train(p, lr):
    optimizer_pre = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weigth_decay)
    net_save_path = 'results/enet_pretrained_%.1f_lr_%.5f.pth' % (float(p), lr)
    try:
        os.mkdir('results')
    except Exception as e:
        print(e)

    labeled_len = int(labeled_data.dataset.imgs.__len__() * float(p))
    labeled_data.dataset.imgs = labeled_data.dataset.imgs[:labeled_len]
    labeled_data.dataset.gts = labeled_data.dataset.gts[:labeled_len]
    print('the length of the labeled dataset is: %d' % labeled_len)
    best_dev_score = -1
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_pre, milestones=[50, 100, 150, 175], gamma=0.5)

    for epoch in range(max_epoch_pre):
        scheduler.step()
        [labeled_score, unlabeled_score, dev_score, validation_score] = [evaluate(net, x, device) for x in
                                                                         (labeled_data, unlabeled_data, dev_data,
                                                                          val_data)]
        for i, (img, mask, _) in tqdm(enumerate(labeled_data)):
            img, mask = img.to(device), mask.to(device)
            optimizer_pre.zero_grad()
            pred = net(img)
            loss = criterion(pred, mask.squeeze(1))
            loss.backward()
            optimizer_pre.step()

        logging.info('pretrained stage: lab:%3f  unlab:%3f  dev:%3f   val:%.3f' % (
            labeled_score, unlabeled_score, dev_score, validation_score))

        score_Dict = {'lab': labeled_score, 'unlab': unlabeled_score, 'val': validation_score, 'dev': dev_score,
                      'epoch': epoch}
        historical_track.append(score_Dict)
        score_Dict.pop('epoch')
        writer.add_scalars('pretrain', score_Dict, epoch)
        pd.DataFrame(historical_track).to_csv(net_save_path.replace('pth', 'csv'))

        # remember best acc@1 and save checkpoint
        is_best = dev_score > best_dev_score  # acc1 > best_acc1
        best_dev_score = max(dev_score, best_dev_score)  # best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': net_save_path.split('_')[0],
            'labeled_score': labeled_score,
            'unlabeled_score': unlabeled_score,
            'validation_score': validation_score,
            'best_dev_score': best_dev_score,
            'state_dict': net.state_dict(),
            'optimizer': optimizer_pre.state_dict(),
            'scheduler': scheduler.state_dict(),
            'labeled_dataloader': labeled_data,
        }, is_best, filename=net_save_path)

    return net_save_path, best_dev_score


def train_baseline(p, net_, net_path_, resume=False, args=None):
    lr = args.baseline_lr
    optim_type = args.optim_type
    config = json.dumps(vars(args))
    saved_path = net_path_.split('/')[0] + '/' + config + '.pth'
    assert optim_type in ('sgd', 'adam')
    if optim_type == 'sgd':
        optimizer_baseline = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weigth_decay)
    else:
        optimizer_baseline = torch.optim.Adam(net.parameters(), lr=lr)

    semi_historical_track = []
    """
    This function performs the training of the pre-trained models with the labeled and unlabeled data.
    """
    global labeled_data
    labeled_len = int(labeled_data.dataset.imgs.__len__() * float(p))
    labeled_data.dataset.imgs = labeled_data.dataset.imgs[:labeled_len]
    labeled_data.dataset.gts = labeled_data.dataset.gts[:labeled_len]
    print('the length of the labeled dataset is: %d' % labeled_len)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_baseline, milestones=[50, 75], gamma=0.5)
    #  loading pre-trained models
    if resume and os.path.isfile(net_path_):
        print("=> loading checkpoint '{}'".format(net_path_))
        checkpoint = torch.load(net_path_, map_location=lambda storage, loc: storage)
        start_epoch = checkpoint['epoch']
        arch_name = checkpoint['arch']
        labeled_score = checkpoint['labeled_score']
        unlabeled_score = checkpoint['unlabeled_score']
        validation_score = checkpoint['validation_score']
        best_dev_score = checkpoint['best_dev_score']
        net_.load_state_dict(checkpoint['state_dict'])

        labeled_data = checkpoint['labeled_dataloader']
        print("=> {} checkpoint of arch '{}' at epoch {}: lab: {:.3f}, unlab: {:.3f}, dev: {:.3f},  val:{:.3f}".format(
            net_path_,
            arch_name,
            start_epoch,
            labeled_score,
            unlabeled_score,
            best_dev_score,
            validation_score))
    else:
        print("=> no checkpoint found at '{}'".format(net_path_))
        raise ValueError
    # net_.load_state_dict(torch.load(net_path_, map_location= map_location)['state_dict'])
    # labeled_data = torch.load(net_path_,map_location=map_location )['labeled_dataloader']
    print('the length of the labeled dataset is: %d' % labeled_data.dataset.imgs.__len__())
    net_.train()

    best_dev_score = -1
    print("STARTING THE BASELINE TRAINING!!!!")
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 130, 160, 180], gamma=0.5)
    for epoch in range(max_epoch_baseline):
        scheduler.step()
        print('epoch = {0:4d}/{1:4d} training baseline'.format(epoch, max_epoch_baseline))

        [labeled_score, unlabeled_score, dev_score, validation_score] = [evaluate(net, x, device) for x in
                                                                         (labeled_data, unlabeled_data, dev_data,
                                                                          val_data)]

        for _ in tqdm(range(int((len(labeled_data) + len(unlabeled_data)) / 2))):  #
            # train with labeled data
            imgs, masks, _ = image_batch_generator(labeled_data, device=device)
            _, llost_list, _ = batch_labeled_loss_(imgs, masks, [net_], criterion)

            # train with unlabeled data
            imgs, _, _ = image_batch_generator(unlabeled_data, device=device)
            pseudolabel, predictions = get_mv_based_labels(imgs, [net_])
            ulost_list = cotraining(predictions, pseudolabel, [net_], criterion, device)
            total_loss = [x + lamda * y for x, y in zip(llost_list, ulost_list)]
            optimizer_baseline.zero_grad()
            total_loss[0].backward()
            optimizer_baseline.step()

        score_Dict = {'lab': labeled_score, 'unlab': unlabeled_score, 'val': validation_score, 'dev': dev_score,
                      'epoch': epoch}
        semi_historical_track.append(score_Dict)
        score_Dict.pop('epoch')
        writer.add_scalars('baseline', score_Dict, epoch)
        pd.DataFrame(semi_historical_track).to_csv(saved_path.replace('pth', 'csv'))

        if dev_score > best_dev_score:
            torch.save(net.state_dict(), saved_path)
            best_dev_score = max(best_dev_score, dev_score)


if __name__ == "__main__":
    # Pre-training Stage
    import argparse

    parser = argparse.ArgumentParser(description='split the training data')
    parser.add_argument('--p', default=0.5, type=float)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pre_lr', default=0.001, type=float)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--baseline_lr', default=0.001, type=float)
    parser.add_argument('--optim_type', default='sgd', choices=('sgd', 'adam'), type=str)
    args = parser.parse_args()
    if args.pretrain:

        saved_path, pretrained_score = pre_train(args.p, lr=args.pre_lr)
        if args.baseline:
            saved_path = saved_path.replace('enet_',
                                            'best_model_')  # 'best_model_'+saved_path  # path corresponding to the best model checkpoint
            train_baseline(args.p, net, saved_path, resume=True, args=args)
    elif bool(args.baseline):
        saved_path = 'results/best_model_' + 'pretrained_%.1f_lr_%.5f.pth' % (args.p, args.pre_lr)
        train_baseline(float(args.p), net, saved_path, resume=True, args=args)
