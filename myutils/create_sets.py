import os
import math
import sys
import shutil
import errno

import numpy as np
import pandas as pd


# TODO: optimize this function
def listFiles(top_dir='.', exten='.jpg'):
    counter = 0
    filesPathList = list()
    for dirpath, dirnames, files in os.walk(top_dir):
        for name in files:
            if name.lower().endswith(exten):
                counter += 1
                filesPathList.append(os.path.join(dirpath, name))

    return filesPathList


def createTrainValSets(root, img_path_list, n_imgs=20, val_portion=0.):
    n_val_imgs = int(round(n_imgs * val_portion)) # number of images in validation set
    print("Number of samples in the val set: ", n_val_imgs)

    if os.path.exists(root):
        train_file = open(os.path.join(root, 'train.csv'), 'w')
        val_file = open(os.path.join(root, 'val.csv'), 'w')

        train_file.write('img,label\n')
        val_file.write('img,label\n')

        for idx, img in enumerate(img_path_list):
            img_name = img.split('/')[-1]

            if idx < n_val_imgs:
                val_file.write("{},{}\n".format(img_name, img_name.replace('.jpg', '_segmentation.png')))
            else:
                train_file.write("{},{}\n".format(img_name, img_name.replace('.jpg', '_segmentation.png')))


def createSemisupervisedSets(root, img_path_list, n_imgs=20, labeled_portion=0.2, test_portion=0.2):
    n_test_imgs = int(math.ceil(n_imgs * test_portion))  # number of images in test set
    n_labeled_imgs = int(math.ceil((n_imgs - n_test_imgs) * labeled_portion))  # number of images in labeled train set

    print("Number of samples in the test set: ", n_test_imgs)
    print("Number of samples in the labeled train set: ", n_labeled_imgs)
    print("Number of samples in the unlabeled train set: ", n_imgs - n_labeled_imgs - n_test_imgs)

    if os.path.exists(root):
        test_file = open(os.path.join(root, 'random_test.csv'), 'w')
        labeled_file = open(os.path.join(root, 'random_labeled_tr.csv'), 'w')
        unlabeled_file = open(os.path.join(root, 'random_unlabeled_tr.csv'), 'w')

        test_file.write('img,label\n')
        labeled_file.write('img,label\n')
        unlabeled_file.write('img,label\n')

        labeled_lim = n_test_imgs + n_labeled_imgs

        for idx, img in enumerate(img_path_list):
            img_name = img.split('/')[-1]

            if idx < n_test_imgs:
                test_file.write("{},{}\n".format(img_name, img_name.replace('.jpg', '_segmentation.png')))
            elif (idx >= n_test_imgs) and (idx < labeled_lim):
                labeled_file.write("{},{}\n".format(img_name, img_name.replace('.jpg', '_segmentation.png')))
            elif idx >= labeled_lim:
                unlabeled_file.write("{},{}\n".format(img_name, img_name.replace('.jpg', '_segmentation.png')))


def split_train_set(output_dir, in_csv_file, train_csv_files, n_splits=3):
    assert len(train_csv_files) == n_splits,\
        "the number files must corrspond with the given number of splits, {}.".format(n_splits)
    img_gts_list = pd.read_csv(in_csv_file)

    n_samples = round(len(img_gts_list) / n_splits)

    # print(img_gts_list['img'][:10])
    for idx in range(n_splits):
        pd.DataFrame(img_gts_list[idx*n_samples:(idx+1)*n_samples]).to_csv(os.path.join(output_dir, train_csv_files[idx]),
                                                                           index=False)


if __name__ == "__main__":
    root_dir = '/home/guillermo/Documents/InternshipETS/SemiSupervisedSegmentation_2/datasets/ISIC2018'

    # img_path_list = listFiles(os.path.join(root_dir, 'ISIC2018_Task1-2_Training_Input'),
    #                                             exten='.jpg')
    # np.random.shuffle(img_path_list)
    # n_imgs = len(img_path_list)
    #
    # # create train and validation sets
    # #createTrainValSets(root_dir, img_path_list, n_imgs=n_imgs, val_portion=0.2)
    #
    # # # create train and validation sets
    # # createLabeledUnlabeledSets(root_dir, img_path_list, n_imgs=n_imgs, labeled_portion=0.3)

    out_dir = os.path.join(root_dir, 'ISIC_Segmenation_dataset_split')
    # createSemisupervisedSets(out_dir, img_path_list, n_imgs=n_imgs, labeled_portion=0.2, test_portion=0.2)

    out_csv_files = [os.path.join(out_dir, 'random_labeled_tr_segnet1.csv'),
                     os.path.join(out_dir, 'random_labeled_tr_segnet2.csv'),
                     os.path.join(out_dir, 'random_labeled_tr_segnet3.csv')]
    train_csv_file = os.path.join(out_dir, 'random_labeled_tr.csv')

    split_train_set(out_dir, train_csv_file, out_csv_files, n_splits=3)




