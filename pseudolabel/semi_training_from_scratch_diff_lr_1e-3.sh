#!/usr/bin/env bash

# =================== learning rate 0.001 (1e-3) ===================

# using oracle loss function
OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=1 python train_onemodel.py --save_dir demo_oracle/semi_from_scratch_diff_lr/lr_0.001/use_oracle --run_semi=True \
--load_pretrain=False --semi_train__lr=0.001 --semi_train__gamma=0.95 --batch_size=4 --semi_train__loss_name=oracle \
--semi_train__update_labeled=True --semi_train__update_unlabeled=True --semi_train__max_epoch=100 --labeled_percentate=0.04 &

# using crossentropy loss function
OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=1 python train_onemodel.py --save_dir demo_oracle/semi_from_scratch_diff_lr/lr_0.001/use_ce --run_semi=True \
--semi_train__lr=0.001 --semi_train__gamma=0.95 --batch_size=4 --semi_train__loss_name=crossentropy \
--semi_train__update_labeled=True --semi_train__update_unlabeled=True --semi_train__max_epoch=100 --labeled_percentate=0.04


