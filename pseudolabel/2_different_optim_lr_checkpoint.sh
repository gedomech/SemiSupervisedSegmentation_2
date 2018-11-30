#!/usr/bin/env bash

## different learning rate
## with adam
CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo_oracle/different_learning_rate/Adamsemi_train__lr_0.0001 --run_semi=True  --model_path runs/demo_oracle/pretrain_lr_0.04/best.pth --semi_train__lr=0.0001 --batch_size=4 --labeled_percentate=0.04 --semi_train__gamma=1 --semi_train__max_epoch=100  &
CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo_oracle/different_learning_rate/Adamsemi_train__lr_0.001 --run_semi=True  --model_path runs/demo_oracle/pretrain_lr_0.04/best.pth --semi_train__lr=0.001 --batch_size=4 --labeled_percentate=0.04 --semi_train__gamma=1 --semi_train__max_epoch=100 &
CUDA_VISIBLE_DEVICES=1 python train_onemodel.py --save_dir demo_oracle/different_learning_rate/Adamsemi_train__lr_0.01 --run_semi=True  --model_path runs/demo_oracle/pretrain_lr_0.04/best.pth --semi_train__lr=0.01 --batch_size=4 --labeled_percentate=0.04 --semi_train__gamma=1 --semi_train__max_epoch=100 &


# with momentum

CUDA_VISIBLE_DEVICES=1 python train_onemodel.py --save_dir demo_oracle/different_learning_rate/SGDMsemi_train__lr_0.0001 --run_semi=True  --model_path runs/demo_oracle/pretrain_lr_0.04/best.pth --semi_train__lr=0.0001 --batch_size=4 --labeled_percentate=0.04 --semi_train__gamma=1  --semi_train__optim_option="{'momentum':0.9}" --semi_train__optim_name=SGD --semi_train__max_epoch=100 &
CUDA_VISIBLE_DEVICES=1 python train_onemodel.py --save_dir demo_oracle/different_learning_rate/SGDMsemi_train__lr_0.001 --run_semi=True  --model_path runs/demo_oracle/pretrain_lr_0.04/best.pth --semi_train__lr=0.001 --batch_size=4 --labeled_percentate=0.04 --semi_train__gamma=1 --semi_train__optim_option="{'momentum':0.9}" --semi_train__optim_name=SGD --semi_train__max_epoch=100 &
CUDA_VISIBLE_DEVICES=2 python train_onemodel.py --save_dir demo_oracle/different_learning_rate/SGDMsemi_train__lr_0.01 --run_semi=True  --model_path runs/demo_oracle/pretrain_lr_0.04/best.pth --semi_train__lr=0.01 --batch_size=4 --labeled_percentate=0.04 --semi_train__gamma=1 --semi_train__optim_option="{'momentum':0.9}" --semi_train__optim_name=SGD --semi_train__max_epoch=100