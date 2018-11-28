#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=3 python train_onemodel.py --save_dir demo/pretrain_lr_0.001 --run_pretrain=True --full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=100

CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo/semi_baseline_last --run_semi=True  --model_path runs/demo/pretrain_lr_0.001/last.pth --semi_train__lr=0.0001 --semi_train__max_epoch=100 &
CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo/semi_baseline_best --run_semi=True  --model_path runs/demo/pretrain_lr_0.001/best.pth --semi_train__lr=0.0001 --semi_train__max_epoch=100