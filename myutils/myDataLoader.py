import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

import warnings
warnings.filterwarnings('ignore')

num_class = 2
spit_ratio = 0.9
means = np.array([0.762824821091, 0.546326646928, 0.570878231817])
stds = np.array([0.0985789149783, 0.0857434017536, 0.0947628491147])

img_transform_std = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=means,
                         std=stds)
])

img_transform_equal = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=means,
    #                      std=stds)
])


mask_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor()
])

class ISICdata(Dataset):
    def __init__(self,root,model,mode,transform,dataAugment=False,equalize=False):
        super().__init__()
        self.dataAugment = dataAugment
        self.equalize = equalize
        self.transform = transform
        assert mode in ('semi','full'), "the mode should be always in 'semi' or 'full', given '%s'."%mode
        self.model = model
        self.mode = mode
        self.root = root
        self.csv_root = os.path.join(root,'ISIC_Segmenation_dataset_split')
        if self.mode=="full":
            assert model in (
            'train', 'dev'), "the model should be always in 'train', 'dev' and 'val', given '%s'." % model
            if self.model=="train":
                img_gts_list = pd.read_csv(os.path.join(self.csv_root,'train.csv'))
            elif self.model=="dev":
                img_gts_list = pd.read_csv(os.path.join(self.csv_root,'val.csv'))
                img_gts_list = img_gts_list.ix[:int(len(img_gts_list) / 2)]
            elif self.model == 'val':
                img_gts_list = pd.read_csv(os.path.join(self.csv_root, 'val.csv'))
                img_gts_list = img_gts_list.ix[int(len(img_gts_list) / 2):]
            else:
                raise NotImplemented
        elif self.mode=="semi":
            assert model in ('labeled', 'unlabeled', 'dev',
                             'val'), "the model should be always in 'labeled', 'unlabeled' or 'test', given '%s'." % model
            if self.model=="labeled":
                img_gts_list = pd.read_csv(os.path.join(self.csv_root,'random_labeled_tr.csv'))
            elif self.model=="unlabeled":
                img_gts_list = pd.read_csv(os.path.join(self.csv_root,'random_unlabeled_tr.csv'))
            elif self.model == "dev":
                img_gts_list = pd.read_csv(os.path.join(self.csv_root, 'random_test.csv'))
                img_gts_list = img_gts_list.ix[:int(len(img_gts_list) / 2)]
            elif self.model == "val":
                img_gts_list = pd.read_csv(os.path.join(self.csv_root, 'random_test.csv'))
                img_gts_list = img_gts_list.ix[int(len(img_gts_list) / 2):]
            else:
                raise NotImplementedError

        self.imgs = img_gts_list['img'].values
        self.gts = img_gts_list['label'].values
        imgs = [x for x in os.listdir(os.path.join(self.root, 'ISIC2018_Task1-2_Training_Input_temp')) if
                x.find('jpg') > 0]
        gts = [x for x in os.listdir(os.path.join(self.root, 'ISIC2018_Task1_Training_GroundTruth_temp')) if
               x.find('png') > 0]
        self.imgs = ['ISIC2018_Task1-2_Training_Input_temp/' + x for x in self.imgs if x.split('/')[0] in imgs]
        self.gts = ['ISIC2018_Task1_Training_GroundTruth_temp/' + x.replace(' ', '') for x in self.gts if
                    x.split('/')[0].replace('_segmentation.png', '.jpg').replace(' ', '') in imgs]

        assert len(self.imgs)==len(self.gts)

    def __getitem__(self, index):
        img_path =self.imgs[index]
        gt_path = self.gts[index]

        img = Image.open(os.path.join(self.root,img_path)).convert('RGB')
        gt = Image.open(os.path.join(self.root,gt_path))

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.dataAugment:
            img, gt = self.augment(img,gt)

        if self.transform:
            img = img_transform_equal(img) if self.equalize else img_transform_std(img)
            gt = mask_transform(gt)

        return img.float(), gt.long(), (img_path,gt_path)

    def augment(self, img, mask):

        if random.random() > 0.2:
            (w, h) = img.size
            (w_, h_) = mask.size
            assert (w==w_ and h==h_),'The size should be the same.'
            crop = random.uniform(0.45, 0.75)
            W = int(crop * w)
            H = int(crop * h)
            start_x = w - W
            start_y = h - H
            x_pos = int(random.uniform(0, start_x))
            y_pos = int(random.uniform(0, start_y))
            img = img.crop((x_pos, y_pos, x_pos + W, y_pos + H))
            mask = mask.crop((x_pos, y_pos, x_pos + W, y_pos + H))

        if random.random() > 0.2:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)

        if random.random() > 0.2:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)

        if random.random() > 0.2:
            angle = random.random() * 90 - 45
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        if random.random() > 0.8 :
            img = img.filter(ImageFilter.GaussianBlur(2))

        if random.random()>0.8:
            img = ImageEnhance.Contrast(img).enhance(1)

        if random.random()>0.8:
            img = ImageEnhance.Brightness(img).enhance(1)

        return img, mask

    def __len__(self):
        return int(len(self.imgs))



if __name__ =="__main__":
    import platform
    from torchnet.meter import AverageValueMeter
    import tqdm
    from torch.utils.data import DataLoader
    if platform.uname()[0]=="windows":
        root ="E:\Applications\QuindiTech\ISICpaperBaseline\datasets\ISIC2018"
    else:
        root = "/home/guillermo/Documents/ETS_ResearchInternship/ISIC2018paperBaseline/datasets/ISIC2018"
    # ISICdata(root=root,model='dev',transform=None)
    traintset = ISICdata(root=root,model='train',transform=True,equalize=True,dataAugment=True)
    train_loader = DataLoader(traintset,batch_size=16,num_workers=8)

    fb_number = AverageValueMeter()
    bg_number = AverageValueMeter()
    fb_number.reset()
    bg_number.reset()

    for i,(img,mask,_) in tqdm.tqdm(enumerate(train_loader)):
        fg = mask.sum()
        bg = (1-mask).sum()
        fb_number.add(fg.item())
        bg_number.add(bg.item())

    print(fb_number.value()[0],bg_number.value()[0])
    print('background/forground_ratio is:',bg_number.value()[0]/fb_number.value()[0] )


