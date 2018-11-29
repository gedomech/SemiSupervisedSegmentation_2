#!/usr/bin/env bash

OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo/oracle/use_oracle --run_semi=True  --model_path runs/demo/pretrain_lr_0.001/best.pth --semi_train__lr=0.0001 --semi_train__gamma=0.95 --batch_size=4 --semi_train__loss_name=oracle
OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo/oracle/use_ce --run_semi=True  --model_path runs/demo/pretrain_lr_0.001/best.pth --semi_train__lr=0.0001 --semi_train__gamma=0.95 --batch_size=4 --semi_train__loss_name=crossentropy

