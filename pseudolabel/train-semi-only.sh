#!/usr/bin/env bash

## different learning rate
## with adam
CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir trainsemi/semi_train__lr_0.0001 --run_semi True --load_pretrain=False  --semi_train__lr=0.0001 --batch_size=4 &
CUDA_VISIBLE_DEVICES=1 python train_onemodel.py --save_dir trainsemi/semi_train__lr_0.001 --run_semi True  --load_pretrain=False  --semi_train__lr=0.001 --batch_size=4 &
CUDA_VISIBLE_DEVICES=2 python train_onemodel.py --save_dir trainsemi/semi_train__lr_0.01 --run_semi True  --load_pretrain=False  --semi_train__lr=0.01 --batch_size=4
