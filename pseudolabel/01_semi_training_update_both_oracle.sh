#!/usr/bin/env bash

OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo_oracle/update_both/use_oracle --run_semi=True --model_path runs/demo_oracle/pretrain_lr_0.04/best.pth --semi_train__lr=0.0001 \
 --semi_train__gamma=0.95 --batch_size=4 --semi_train__loss_name=oracle --semi_train__update_labeled=True --semi_train__update_unlabeled=True --semi_train__max_epoch=100 --labeled_percentate=0.04
