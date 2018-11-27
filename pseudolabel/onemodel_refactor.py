# coding=utf-8
import logging
import sys
from abc import ABC, abstractmethod

from absl import flags, app
from tensorboardX import SummaryWriter

import pseudolabel.mask_gene
from myutils.myDataLoader import ISICdata
from myutils.myENet import Enet
from myutils.myLoss import CrossEntropyLoss2d
from myutils.myUtils import *

logging.basicConfig(format='%(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
sys.path.extend([os.path.dirname(os.getcwd())])
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings('ignore')


def get_dataloader(hparam):
    root = "../datasets/ISIC2018"
    labeled_data = ISICdata(root=root, model='labeled', mode='semi', transform=True,
                            dataAugment=False, equalize=False)
    unlabeled_data = ISICdata(root=root, model='unlabeled', mode='semi', transform=True,
                              dataAugment=False, equalize=False)
    val_data = ISICdata(root=root, model='val', mode='semi', transform=True,
                        dataAugment=False, equalize=False)

    labeled_loader_params = {'batch_size': hparam['num_workers'],
                             'shuffle': True,  # True
                             'num_workers': hparam['batch_size'],
                             'pin_memory': True}

    unlabeled_loader_params = {'batch_size': hparam['num_workers'],
                               'shuffle': True,  # True
                               'num_workers': hparam['batch_size'],
                               'pin_memory': True}
    val_loader_params = {'batch_size': hparam['num_workers'],
                         'shuffle': False,  # True
                         'num_workers': hparam['batch_size'],
                         'pin_memory': True}

    labeled_data = DataLoader(labeled_data, **labeled_loader_params)
    unlabeled_data = DataLoader(unlabeled_data, **unlabeled_loader_params)
    val_data = DataLoader(val_data, **val_loader_params)
    return {'labeled': labeled_data,
            'unlabeled': unlabeled_data,
            'val': val_data}


class Trainer(ABC):
    def __init__(self, torchnet) -> None:
        super().__init__()
        self.torchnet = torchnet

    @abstractmethod
    def _train(self, **kwargs):
        pass

    def set_writer(self, writer):
        self.writer = writer

    def _evaluate(self, dataloader):
        dice_meter = AverageValueMeter()
        self.torchnet.eval()
        with torch.no_grad():
            for i, (img, gt, _) in enumerate(dataloader):
                img, gt = img.to(device), gt.to(device)
                pred_logit = self.torchnet(img)
                pred_mask = pred2segmentation(pred_logit)
                dice_meter.add(dice_loss(pred_mask, gt))
        self.torchnet.train()
        return dice_meter.value()[0]

    @property
    def save_dict(self):
        return self.torchnet.state_dict()

    @classmethod
    def _rm_alias(cls, hparam):
        new_hparam = {}
        for k, v in hparam.items():
            if k.find(cls.alias) >= 0:
                new_hparam[k.replace(cls.alias, '')] = v
        return new_hparam

    def checkpoint(self, dice, epoch, name=None):
        try:
            getattr(self, 'best_dice')
        except:
            self.best_dice = -1

        if dice > self.best_dice:
            self.best_dice = dice
            dict2save = dict()
            dict2save['epoch'] = epoch
            dict2save['dice'] = dice
            dict2save['model'] = self.save_dict
            if name is None:
                torch.save(dict2save, 'best.pth')
            else:
                torch.save(dict2save, name)
        else:
            return


class FullysupervisedTrainer(Trainer):
    alias = 'full_train__'
    lrscheduler_keys = ['milestones', 'gamma']
    optim_keys = ['lr', 'weight_decay', 'amsgrad']
    criterion_keys = ['weight']

    @classmethod
    def set_flag(cls):
        flags.DEFINE_integer(cls.alias + 'max_epoch', default=200, help='max_epoch for full training')
        flags.DEFINE_multi_integer(cls.alias + 'milestones', default=[20, 40, 60, 80, 100, 120, 140, 160, 180],
                                   help='milestones for full training')
        flags.DEFINE_float(cls.alias + 'gamma', default=0.5, help='gamma for lr_scheduler in full training')
        flags.DEFINE_float(cls.alias + 'lr', default=0.001, help='lr for full training')
        flags.DEFINE_float(cls.alias + 'weight_decay', default=0, help='weight_decay for full training')
        flags.DEFINE_multi_float(cls.alias + 'weight', default=[1, 1], help='weight balance for CE for full training')
        flags.DEFINE_string(cls.alias + 'loss_name', default='crossentropy', help='criterion used in the full training')
        flags.DEFINE_string(cls.alias + 'optim', default='Adam', help='optimzer used in the full training')
        flags.DEFINE_string(cls.alias + 'scheduler', default='MultiStepLR', help='scheduler used in the full training')

    def __init__(self, torchnet, dataloader, hparam) -> None:
        super().__init__(torchnet)
        self.name = 'Fully_Supervised Training'
        self.dataloader = dataloader
        self.hparam = copy.deepcopy(self._rm_alias(hparam))
        optim_hparam = self._rm_alias(extract_from_big_dict(self.hparam, FullysupervisedTrainer.optim_keys))
        self.optim = getattr(torch.optim, self.hparam['optim'])(self.torchnet.parameters(), **optim_hparam)
        lrschduler_hparam = extract_from_big_dict(self.hparam, FullysupervisedTrainer.lrscheduler_keys)
        self.lrscheduler = getattr(torch.optim.lr_scheduler, self.hparam['scheduler'])(self.optim, **lrschduler_hparam)
        criterion_hparam = extract_from_big_dict(self.hparam, FullysupervisedTrainer.criterion_keys)
        self.criterion = get_citerion(self.hparam['loss_name'], **criterion_hparam)
        self.criterion.to(device)

    def start_training(self):
        logger.info(self.name + '  Training starts:')
        for epoch in range(self.hparam['max_epoch']):
            self.lrscheduler.step()
            self._train(self.dataloader['labeled'], self.criterion)
            with torch.no_grad():
                dice = self._evaluate(self.dataloader['labeled'])
                logger.info('labeled_data dice: {.3f} at epoch: {}'.format(dice, epoch))
                self.writer.add_scalar('%s/labeled' % self.name, dice, epoch)
                dice = self._evaluate(self.dataloader['unlabeled'])
                logger.info('unlabeled_data dice: {.3f} at epoch: {}'.format(dice, epoch))
                self.writer.add_scalar('%s/unlabeled' % self.name, dice, epoch)
                dice = self._evaluate(self.dataloader['val'])
                logger.info('val_data dice: {.3f} at epoch: {}'.format(dice, epoch))
                self.writer.add_scalar('%s/val' % self.name, dice, epoch)

    def _train(self, dataloader, criterion):
        for i, (img, gt, _) in enumerate(dataloader):
            self.optim.zero_grad()
            self.torchnet.eval()
            img, gt = img.to(device), gt.to(device)
            pred_logit = self.torchnet(img)
            loss = criterion(pred_logit, gt.squeeze(1))
            loss.backward()
            self.optim.step()


class SemisupervisedTrainer(Trainer):
    alias = 'semi_train__'
    lrscheduler_keys = ['milestones', 'gamma']
    optim_keys = ['lr', 'weight_decay', 'amsgrad']
    flow_control_keys = ['update_labeled', 'update_unlabeled']
    criterion_keys = ['weight']

    @classmethod
    def set_flag(cls):
        flags.DEFINE_integer(cls.alias + 'max_epoch', default=200, help='max_epoch for semi training')
        flags.DEFINE_multi_integer(cls.alias + 'milestones', default=[20, 40, 60, 80, 100, 120, 140, 160, 180],
                                   help='milestones for semi training')
        flags.DEFINE_float(cls.alias + 'gamma', default=0.8, help='gamma for lr_scheduler in semi training')
        flags.DEFINE_float(cls.alias + 'lr', default=0.001, help='lr for semi training')
        flags.DEFINE_float(cls.alias + 'weight_decay', default=0, help='weight_decay for semi training')
        flags.DEFINE_multi_float(cls.alias + 'weight', default=[1, 1], help='weight balance for CE for semi training')
        flags.DEFINE_string(cls.alias + 'loss_name', default='crossentropy', help='criterion used in the semi training')
        flags.DEFINE_string(cls.alias + 'optim', default='Adam', help='optimzer used in the semi training')
        flags.DEFINE_string(cls.alias + 'scheduler', default='MultiStepLR', help='scheduler used in the full training')

    def __init__(self, torchnet, dataloader, hparam) -> None:
        super().__init__(torchnet)
        self.name = 'Semi_Supervised Training'
        self.dataloader = dataloader
        self.hparam = copy.deepcopy(self._rm_alias(hparam))
        optim_hparam = extract_from_big_dict(self.hparam, SemisupervisedTrainer.optim_keys)
        self.optim = getattr(torch.optim, self.hparam['optim'])(self.torchnet.parameters(), **optim_hparam)
        lrschduler_hparam = extract_from_big_dict(self.hparam, SemisupervisedTrainer.lrscheduler_keys)
        self.lrscheduler = getattr(torch.optim.lr_scheduler, self.hparam['scheduler'])(self.optim,
                                                                                       **lrschduler_hparam)
        self.mask_generation = getattr(pseudolabel.mask_gene, 'naiveway')
        criterion_hparam = extract_from_big_dict(self.hparam, SemisupervisedTrainer.criterion_keys)
        self.criterion = get_citerion(self.hparam['loss_name'], **criterion_hparam)
        self.criterion.to(device)

    def start_training(self):
        for epoch in range(self.hparam['max_epoch']):
            self.lrscheduler.step()
            self._train(self.dataloader['labeled'], self.dataloader['unlabeled'], self.criterion)
            with torch.no_grad():
                dice = self._evaluate(self.dataloader['labeled'])
                logger.info('labeled_data dice: {.3f} at epoch: {}'.format(dice, epoch))
                self.writer.add_scalar('%s/labeled' % self.name, dice, epoch)
                dice = self._evaluate(self.dataloader['unlabeled'])
                logger.info('unlabeled_data dice: {.3f} at epoch: {}'.format(dice, epoch))
                self.writer.add_scalar('%s/unlabeled' % self.name, dice, epoch)
                dice = self._evaluate(self.dataloader['val'])
                logger.info('val_data dice: {.3f} at epoch: {}'.format(dice, epoch))
                self.writer.add_scalar('%s/val' % self.name, dice, epoch)

    def _train(self, l_dataloader, u_dataloader, criterion):
        for i, ((limg, lgt, _), (uimg, ugt, _)) in enumerate(zip(l_dataloader, u_dataloader)):
            if self.hparam['update_labeled']:
                self.optim.zero_grad()
                limg, lgt = limg.to(device), lgt.to(device)
                pred_logit = self.torchnet(limg)
                loss = criterion(pred_logit, lgt.squeeze(1))
                loss.backward()
                self.optim.step()
            if self.hparam['update_unlabeled']:
                self.optim.zero_grad()
                uimg, ugt = uimg.to(device), ugt.to(device)
                pred_logit = self.torchnet(uimg)
                mask = self.mask_generation(pred_logit)
                loss = criterion(pred_logit, mask)
                loss.backward()
                self.optim.step()


def get_default_parameter():
    flags.DEFINE_integer('num_workers', default=4, help='number of workers used in dataloader')
    flags.DEFINE_integer('batch_size', default=4, help='number of batch size')
    flags.DEFINE_boolean('semi_train__update_labeled', default=True,
                         help='update the labeled image while self training')
    flags.DEFINE_boolean('semi_train__update_unlabeled', default=True,
                         help='update the unlabeled image while self training')
    flags.DEFINE_boolean('run_pretrain', default=True,
                         help='run_pretrain')
    flags.DEFINE_boolean('run_semi', default=True,
                         help='run_self_training')


def get_citerion(lossname, **kwargs):
    if lossname == 'crossentropy':
        criterion = CrossEntropyLoss2d(**kwargs)
    else:
        raise NotImplementedError
    return criterion


class TrainWrapper(ABC):

    def __init__(self, fullTrainer: FullysupervisedTrainer, semiTrainer: SemisupervisedTrainer) -> None:
        super().__init__()
        self.fulltrainer = fullTrainer
        self.semitrainer = semiTrainer
        self.writer = SummaryWriter('runs/' + self.writer_name)
        self.fulltrainer.set_writer(self.writer)
        self.semitrainer.set_writer(self.writer)

    def run_fully_training(self):
        self.fulltrainer.start_training()

    def run_semi_training(self):
        self.semitrainer.start_training()

    @property
    def writer_name(self):
        return self.generate_random_str() + '_' + self.generate_current_time()

    @staticmethod
    def generate_random_str(randomlength=16):
        """
        生成一个指定长度的随机字符串
        """
        import random
        random_str = ''
        base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
        length = len(base_str) - 1
        for i in range(randomlength):
            random_str += base_str[random.randint(0, length)]
        return random_str

    @staticmethod
    def generate_current_time():
        from time import strftime, localtime
        ctime = strftime("%Y-%m-%d %H:%M:%S", localtime())
        return ctime

    @classmethod
    def save_hparams(hparams, writer_name):
        import pandas as pd
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(hparams.items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        file_name = os.path.join(writer_name, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        pd.Series(hparams).to_csv(os.path.join(writer_name, 'opt.csv'))


def run(argv):
    del argv

    hparam = flags.FLAGS.flag_values_dict()
    ## data for semi-supervised training
    dataloaders = get_dataloader(hparam)

    ## networks and optimisers
    net = Enet(2)
    net = net.to(device)

    fullytrainer = FullysupervisedTrainer(net, dataloaders, hparam)
    semitrainer = SemisupervisedTrainer(net, dataloaders, hparam)
    trainWrapper = TrainWrapper(fullytrainer, semitrainer)
    if hparam['run_pretrain']:
        trainWrapper.run_fully_training()
    if hparam['run_semi']:
        trainWrapper.run_semi_training()


if __name__ == '__main__':
    FullysupervisedTrainer.set_flag()
    SemisupervisedTrainer.set_flag()
    get_default_parameter()
    app.run(run)
