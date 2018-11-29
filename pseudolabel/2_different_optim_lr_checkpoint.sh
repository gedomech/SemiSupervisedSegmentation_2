#!/usr/bin/env bash

## different learning rate
## with adam
#CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo/different_learning_rate/Adamsemi_train__lr_0.0001 --run_semi True  --model_path runs/demo/pretrain_lr_0.001/best.pth --semi_train__lr=0.0001 --batch_size=4 &
#CUDA_VISIBLE_DEVICES=1 python train_onemodel.py --save_dir demo/different_learning_rate/Adamsemi_train__lr_0.001 --run_semi True  --model_path runs/demo/pretrain_lr_0.001/best.pth --semi_train__lr=0.001 --batch_size=4 &
#CUDA_VISIBLE_DEVICES=2 python train_onemodel.py --save_dir demo/different_learning_rate/Adamsemi_train__lr_0.01 --run_semi True  --model_path runs/demo/pretrain_lr_0.001/best.pth --semi_train__lr=0.01 --batch_size=4
# with SGD

OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo/different_learning_rate/SGDsemi_train__lr_0.0001 --run_semi True  --model_path runs/demo/pretrain_lr_0.001/best.pth --semi_train__lr=0.0001 --semi_train__optim_name=SGD  --batch_size=4 &
OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=1 python train_onemodel.py --save_dir demo/different_learning_rate/SGDsemi_train__lr_0.001 --run_semi True  --model_path runs/demo/pretrain_lr_0.001/best.pth --semi_train__lr=0.001 --semi_train__optim_name=SGD  --batch_size=4 &
OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo/different_learning_rate/SGDsemi_train__lr_0.01 --run_semi True  --model_path runs/demo/pretrain_lr_0.001/best.pth --semi_train__lr=0.01 --semi_train__optim_name=SGD  --batch_size=4

# with momentum
#
OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo/different_learning_rate/SGDMsemi_train__lr_0.0001 --run_semi True  --model_path runs/demo/pretrain_lr_0.001/best.pth --semi_train__lr=0.0001 --semi_train__optim_name=SGD  --batch_size=4 --semi_train__optim_option="{'momentum':0.9}" &
OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=1 python train_onemodel.py --save_dir demo/different_learning_rate/SGDMsemi_train__lr_0.001 --run_semi True  --model_path runs/demo/pretrain_lr_0.001/best.pth --semi_train__lr=0.001 --semi_train__optim_name=SGD  --batch_size=4 --semi_train__optim_option="{'momentum':0.9}" &
OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo/different_learning_rate/SGDMsemi_train__lr_0.01 --run_semi True  --model_path runs/demo/pretrain_lr_0.001/best.pth --semi_train__lr=0.01 --semi_train__optim_name=SGD  --batch_size=4 --semi_train__optim_option="{'momentum':0.9}"