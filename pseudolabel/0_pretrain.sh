#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo_oracle/pretrain_lr_0.1 --run_pretrain=True --full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=200 --labeled_percentate=0.1

CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo_oracle/pretrain_lr_0.3 --run_pretrain=True --full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=200 --labeled_percentate=0.3

CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo_oracle/pretrain_lr_0.5 --run_pretrain=True --full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=200 --labeled_percentate=0.5

CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo_orcale/pretrain_lr_0.7 --run_pretrain=True --full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=200 --labeled_percentate=0.7
CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo_orcale/pretrain_lr_1  --run_pretrain=True --full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=200 --labeled_percentate=1.0
