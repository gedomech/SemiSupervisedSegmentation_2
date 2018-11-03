import os
import sys
import pandas as pd
from tensorboardX import SummaryWriter

sys.path.extend([os.path.dirname(os.getcwd())])
from myutils.myDataLoader import ISICdata
from myutils.myENet import Enet
from myutils.myNetworks import UNet, SegNet
from myutils.myLoss import CrossEntropyLoss2d
import warnings
from tqdm import tqdm
from myutils.myUtils import *

import csv
import argparse

warnings.filterwarnings('ignore')
writer = SummaryWriter()

# torch.set_num_threads(1)  # set by deafault to 1
root = "datasets/ISIC2018"
writer = SummaryWriter()

class_number = 2
lr = 1e-5
weigth_decay = 1e-6
lamda = 5e-2

use_cuda = True
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
number_workers = 0
labeled_batch_size = 1
unlabeled_batch_size = 1
val_batch_size = 1

max_epoch_pre = 1
max_epoch_baseline = 2
max_epoch_ensemble = 100
train_print_frequncy = 10
val_print_frequncy = 10

Equalize = False
# data for semi-supervised training
labeled_data = ISICdata(root=root, model='labeled', mode='semi', transform=True,
                        dataAugment=False, equalize=Equalize)

# customizing labeled training sets for SegNets
labeled_data_Segnet1 = ISICdata(root=root, model='labeled', mode='customized', transform=True,
                                img_gts_file='random_labeled_tr_segnet1.csv', dataAugment=False, equalize=Equalize)
labeled_data_Segnet2 = ISICdata(root=root, model='labeled', mode='customized', transform=True,
                                img_gts_file='random_labeled_tr_segnet2.csv', dataAugment=False, equalize=Equalize)
labeled_data_Segnet3 = ISICdata(root=root, model='labeled', mode='customized', transform=True,
                                img_gts_file='random_labeled_tr_segnet3.csv', dataAugment=False, equalize=Equalize)

unlabeled_data = ISICdata(root=root, model='unlabeled', mode='semi', transform=True,
                          dataAugment=False, equalize=Equalize)
test_data = ISICdata(root=root, model='test', mode='semi', transform=True,
                     dataAugment=False, equalize=Equalize)

labeled_loader_params = {'batch_size': labeled_batch_size,
                         'shuffle': True,
                         'num_workers': number_workers,
                         'pin_memory': True}

unlabeled_loader_params = {'batch_size': unlabeled_batch_size,
                           'shuffle': True,
                           'num_workers': number_workers,
                           'pin_memory': True}

# customizing labeled training dataloaders for SegNets
labeled_data_Segnet1 = DataLoader(labeled_data_Segnet1, **labeled_loader_params)
labeled_data_Segnet2 = DataLoader(labeled_data_Segnet2, **labeled_loader_params)
labeled_data_Segnet3 = DataLoader(labeled_data_Segnet3, **labeled_loader_params)

unlabeled_data = DataLoader(unlabeled_data, **unlabeled_loader_params)
test_data = DataLoader(test_data, **unlabeled_loader_params)

## networks and optimisers
# nets = [Enet(class_number),
#         #UNet(class_number),
#         SegNet(class_number)]

nets = [SegNet(class_number),
        SegNet(class_number),
        SegNet(class_number)]

nets = map_(lambda x: x.to(device), nets)

map_location = lambda storage, loc: storage

optimizers = [torch.optim.Adam(nets[0].parameters(), lr=lr, weight_decay=weigth_decay),
              torch.optim.Adam(nets[1].parameters(), lr=lr, weight_decay=weigth_decay),
              torch.optim.Adam(nets[2].parameters(), lr=lr, weight_decay=weigth_decay)]

## loss
class_weigth = [1 * 0.1, 3.53]
class_weigth = torch.Tensor(class_weigth)
criterion = CrossEntropyLoss2d(class_weigth).to(device) if (
        torch.cuda.is_available() and use_cuda) else CrossEntropyLoss2d(
    class_weigth)
ensemble_criterion = JensenShannonDivergence(reduce=True, size_average=False)

nets_names = ['segnet1', 'segnet2', 'segnet3']
historical_score_dict = {
    'epoch': -1,
    nets_names[0]: 0,
    nets_names[1]: 0,
    nets_names[2]: 0,
    'mv': 0,
    'jsd': 0}

from functools import partial


def pre_train():
    """
    This function performs the training with the unlabeled images.
    """
    map_(lambda x: x.train(), nets)

    global historical_score_dict

    nets_path = ['checkpoint/best_ENet_pre-trained.pth',
                 'checkpoint/best_UNet_pre-trained.pth',
                 'checkpoint/best_SegNet_pre-trained.pth']
    dice_meters = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]

    from multiprocessing import Pool

    for epoch in range(max_epoch_pre):
        map_(lambda x: x.reset(), dice_meters)

        if epoch % 5 == 0:
            learning_rate_decay(optimizers, 0.95)

        for i, (img, mask, _) in tqdm(enumerate(labeled_data)):
            img, mask = img.to(device), mask.to(device)

            p_forward = partial(s_forward_backward, imgs=img, masks=mask, criterion=criterion)
            dices = list(Pool().starmap(p_forward, zip(nets, optimizers)))
            map_(lambda x, y: x.add(y), dice_meters, dices)

        print(
            'traing epoch {0:1d}/{1:d} pre-training: enet_dice_score: {2:.6f}, unet_dice_score: {3:.6f}, segnet_dice_score: {4:.6f}'.format(
                epoch + 1, max_epoch_pre, dice_meters[0].value()[0],
                dice_meters[1].value()[0], dice_meters[2].value()[0]))

        score_meters, ensemble_score = test(nets, test_data, device=device)

        print(
            'val epoch {0:d}/{1:d} pre-training: enet_dice_score: {2:.6f}, unet_dice_score: {3:.6f}, segnet_dice_score: {4:.6f}, with majorty voting: {5:.6f}'.format(
                epoch + 1,
                max_epoch_pre,
                score_meters[0].value()[0].item(),
                score_meters[1].value()[0].item(),
                score_meters[2].value()[0].item(),
                ensemble_score.value()[0]))

        historical_score_dict = save_models(nets, nets_path, score_meters, epoch, historical_score_dict)

        # visualize(writer, nets_, unlabeled_loader_, 8, epoch, randomly=False)


def train_baseline(nets_, nets_path_, labeled_loader_: list, unlabeled_loader_, cvs_writer):
    records =[]
    """
    This function performs the training of the pre-trained models with the labeled and unlabeled data.
    """
    #  loading pre-trained models
    map_(lambda x, y: [x.load_state_dict(torch.load(y, map_location='cpu')), x.train()], nets_, nets_path_)
    global historical_score_dict
    nets_path = ['checkpoint/best_SegNet1_baseline_test.pth',
                 'checkpoint/best_SegNet2_baseline_test.pth',
                 'checkpoint/best_SegNet3_baseline_test.pth']
    dice_meters = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]

    # registering the initial performance of the pre-trained networks
    score_meters, ensemble_score = test(nets_, test_data, device=device)

    # add performance of nets to plot
    nets_score_dict = {"Segnet1": score_meters[0].value()[0].item(),
                       "Segnet2": score_meters[1].value()[0].item(),
                       "Segnet3": score_meters[2].value()[0].item(),
                       "MajVote": ensemble_score.value()[0]}
    add_visual_perform(writer, nets_score_dict, 0)

    print("STARTING THE BASELINE TRAINING!!!!")
    for epoch in range(max_epoch_baseline):
        print('epoch = {0:4d}/{1:4d} training baseline'.format(epoch, max_epoch_baseline))

        # if epoch % 5 == 0:
        #     learning_rate_decay(optimizers, 0.95)

        # train with labeled data
        for _ in tqdm(range(max(len(labeled_loader_[0]), len(unlabeled_loader_)))):  # I need to optimize this

            _, llost_list, dice_score = batch_labeled_loss_customized(labeled_loader_, device, nets_, criterion)
            # _, llost_list, dice_score = batch_labeled_loss_(imgs, masks, nets_, criterion)

            # train with unlabeled data
            imgs, _, _ = image_batch_generator(unlabeled_loader_, device=device)
            pseudolabel, predictions = get_mv_based_labels(imgs, nets_)
            ulost_list = cotraining(predictions, pseudolabel, nets_, criterion,device)
            total_loss = [x + lamda * y for x, y in zip(llost_list, ulost_list)]

            for idx in range(len(optimizers)):
                optimizers[idx].zero_grad()
                total_loss[idx].backward()
                optimizers[idx].step()
                dice_meters[idx].add(dice_score[idx])

        print(
            'train epoch {0:1d}/{1:d} baseline: segnet1_dice_score={2:.6f}, segnet2_dice_score={3:.6f}, segnet3_dice_score={4:.6f}'.format(
                epoch + 1, max_epoch_baseline, dice_meters[0].value()[0].item(),
                dice_meters[1].value()[0].item(), dice_meters[2].value()[0].item()))

        score_meters, ensemble_score = test(nets_, test_data, device=device)

        # add performance of nets to plot
        nets_score_dict = {"Segnet1": score_meters[0].value()[0].item(),
                           "Segnet2": score_meters[1].value()[0].item(),
                           "Segnet3": score_meters[2].value()[0].item(),
                           "MajVote": ensemble_score.value()[0]}
        add_visual_perform(writer, nets_score_dict, epoch+1)

        print(
            'val epoch {0:d}/{1:d} baseline: segnet1_dice_score={2:.6f}, segnet2_dice_score={3:.6f}, segnet3_dice_score={4:.6f}, with majorty_voting={5:.6f}'.format(
                epoch + 1,
                max_epoch_baseline,
                score_meters[0].value()[0].item(),
                score_meters[1].value()[0].item(),
                score_meters[2].value()[0].item(),
                ensemble_score.value()[0]))

        cvs_writer.writerow({'Epoch': epoch + 1,
                             'SegNet1_Score': score_meters[0].value()[0].item(),
                             'SegNet2_Score': score_meters[1].value()[0].item(),
                             'SegNet3_Score': score_meters[2].value()[0].item(),
                             'MV_Score': ensemble_score.value()[0]})
        rec_data = {'Epoch': epoch + 1,
                    'SegNet1_Score': score_meters[0].value()[0].item(),
                    'SegNet2_Score': score_meters[1].value()[0].item(),
                    'SegNet3_Score': score_meters[2].value()[0].item(),
                    'MV_Score': ensemble_score.value()[0]}
        try:
            pd.DataFrame([rec_data]).to_csv('output_baseline_01112018_Segnet.csv', index=False, float_format='%.4f')
        except Exception as e:
            print(e)

        historical_score_dict = save_models(nets_, nets_path, nets_names, score_meters, epoch, historical_score_dict)
        if ensemble_score.value()[0] > historical_score_dict['mv']:
            historical_score_dict['mv'] = ensemble_score.value()[0]

            records.append(historical_score_dict)

            try:
                pd.DataFrame(records).to_csv('baseline_records_segnet_ens_test.csv', index=False, float_format='%.4f')
            except Exception as e:
                print(e)

        # visualize(writer, nets_, unlabeled_loader_, 8, epoch, randomly=False)


def train_ensemble(nets_, nets_path_, labeled_loader_, unlabeled_loader_, cvs_writer):
    """
    train_ensemble function performs the ensemble training with the unlabeled subset.
    """
    records = []
    global historical_score_dict
    map_(lambda x, y: [x.load_state_dict(torch.load(y, map_location='cpu')), x.train()], nets_, nets_path_)

    # nets_path = ['checkpoint/best_ENet_ensemble.pth',
    #              # 'checkpoint/best_UNet_ensemble.pth',
    #              'checkpoint/best_SegNet_ensemble.pth']
    nets_path = ['checkpoint/best_SegNet1_pre-trained.pth',
                 'checkpoint/best_SegNet2_pre-trained.pth',
                 'checkpoint/best_SegNet3_pre-trained.pth']

    dice_meters = [AverageValueMeter(), AverageValueMeter(), AverageValueMeter()]

    for epoch in range(max_epoch_ensemble):
        print('epoch = {0:4d}/{1:4d}'.format(epoch, max_epoch_ensemble))
        for idx, _ in enumerate(nets_):
            dice_meters[idx].reset()

        # if epoch % 5 == 0:
        #     learning_rate_decay(optimizers, 0.95)

        for _ in tqdm(range(max(len(labeled_loader_), len(unlabeled_loader_)))):  # max(len(labeled_loader_), len(unlabeled_loader_)
            # === train with labeled data ===
            imgs, masks, _ = image_batch_generator(labeled_loader_, device=device)
            _, llost_list, dice_score = batch_labeled_loss_(imgs, masks, nets_, criterion)

            # train with unlabeled data
            imgs, _, _ = image_batch_generator(unlabeled_loader_, device=device)
            pseudolabel, predictions = get_mv_based_labels(imgs, nets_)
            jsdLoss = get_loss(predictions)

            total_loss = [x + jsdLoss for x in llost_list]
            for idx, optim in enumerate(optimizers):
                optim.zero_grad()
                total_loss[idx].backward(retain_graph=True)
                optim.step()
                dice_meters[idx].add(dice_score[idx])

        print(
            'train epoch {0:1d}/{1:d} baseline: enet_dice_score={2:.6f}, segnet_dice_score={3:.6f}'.format(
                epoch + 1, max_epoch_pre, dice_meters[0].value()[0],
                dice_meters[1].value()[0], dice_meters[2].value()[0]))

        score_meters, ensemble_score = test(nets_, test_data, device=device)

        print(
            'val epoch {0:d}/{1:d} baseline: segnet1_dice_score={2:.6f}, segnet2_dice_score={3:.6f}, segnet3_dice_score={4:.6f}, with majorty_voting={5:.6f}'.format(
                epoch + 1,
                max_epoch_pre,
                score_meters[0].value()[0].item(),
                score_meters[1].value()[0].item(),
                score_meters[2].value()[0].item(),
                ensemble_score.value()[0]))

        cvs_writer.writerow({'Epoch': epoch + 1,
                             'ENet_Score': score_meters[0].value()[0].item(),
                             'SegNet_Score': score_meters[1].value()[0].item(),
                             'MV_Score': ensemble_score.value()[0]})

        historical_score_dict = save_models(nets_, nets_path, score_meters, epoch, historical_score_dict)
        if ensemble_score.value()[0] > historical_score_dict['jsd']:
            historical_score_dict['jsd'] = ensemble_score.value()[0]

        records.append(historical_score_dict)

        try:
            pd.DataFrame(records).to_csv('ensemblerecords.csv')
        except Exception as e:
            print(e)

        # visualize(writer, nets_, unlabeled_loader_, 8, epoch, randomly=False)


if __name__ == "__main__":
    PRE_TRAINING = False
    BASELINE = False
    ENSEMBLE = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-training", type=str2bool, nargs='?', const=False, default=PRE_TRAINING,
                        help="Whether to pre-train the models.")
    parser.add_argument("--baseline", type=str2bool, nargs='?', const=False, default=BASELINE,
                        help="Whether to train the baseline models.")
    parser.add_argument("--ensemble", type=str2bool, nargs='?', const=False, default=ENSEMBLE,
                        help="Whether to train the ensemble models.")

    args = parser.parse_args()

    # nets_path_ = ['checkpoint/best_ENet_pre-trained.pth',
    #               # 'checkpoint/best_UNet_pre-trained.pth',
    #               'checkpoint/best_SegNet_pre-trained.pth']
    nets_path_ = 3*['checkpoint/best_SegNet_pre-trained.pth']

    if args.pre_training:
        # Pre-training Stage
        print('STARTING THE PRE-TRAINING STAGE')
        pre_train()
    elif args.baseline:
        # Baseline Training Stage
        print('STARTING THE BASELINE TRAINING STAGE')
        baseline_file = open('output_baseline_01112018_Segnet_test_outside.csv', 'w')
        # baseline_fields = ['Epoch', 'ENet_Score', 'SegNet_Score', 'MV_Score']
        baseline_fields = ['Epoch', 'SegNet1_Score', 'SegNet2_Score', 'SegNet3_Score', 'MV_Score']
        baseline_writer = csv.DictWriter(baseline_file, fieldnames=baseline_fields)
        baseline_writer.writeheader()
        train_baseline(nets,
                       nets_path_,
                       [labeled_data_Segnet1, labeled_data_Segnet2, labeled_data_Segnet3],
                       unlabeled_data,
                       baseline_writer)
        # train_baseline(nets, nets_path_, labeled_data, unlabeled_data, baseline_writer)
    elif args.ensemble:
        # Ensemble Training Stage
        print('STARTING THE ENSEMBLE TRAINING STAGE')
        ensemble_file = open('output_ensemble_31102018.csv', 'w')
        ensemble_fields = ['Epoch', 'ENet_Score', 'SegNet_Score', 'MV_Score']
        ensemble_writer = csv.DictWriter(ensemble_file, fieldnames=ensemble_fields)
        ensemble_writer.writeheader()
        # train_ensemble(nets, nets_path_, labeled_data, unlabeled_data, ensemble_writer)

    # baseline_writer = None
    # nets_path_ = 3 * ['checkpoint/best_SegNet_pre-trained.pth']
    # train_baseline(nets,
    #                nets_path_,
    #                [labeled_data_Segnet1, labeled_data_Segnet2, labeled_data_Segnet3],
    #                unlabeled_data,
    #                baseline_writer)

