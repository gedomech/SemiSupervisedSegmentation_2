#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir pretrain_lr_0.01 --run_pretrain True --full_train__gamma=0.5 --full_train__lr=0.01 &
CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir pretrain_lr_0.001 --run_pretrain True --full_train__gamma=0.8 --full_train__lr=0.001


## take pretrain_lr_0.01 for example
CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir semi_baseline --run_semi True  --model_path runs/pretrain_lr_0.01/best.pth &
CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir semi_without_labeleddata --run_semi True --semi_train__update_labeled=False --model_path runs/pretrain_lr_0.01/best.pth


# compared with different batch size
CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir semi_batch_1 --run_semi True  --model_path runs/pretrain_lr_0.01/best.pth --batch_size 1

CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir semi_batch_8 --run_semi True  --model_path runs/pretrain_lr_0.01/best.pth --batch_size 8

CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir semi_batch_16 --run_semi True  --model_path runs/pretrain_lr_0.01/best.pth --batch_size 16



