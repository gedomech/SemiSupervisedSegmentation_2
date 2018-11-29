#!/usr/bin/env bash
#!/usr/bin/env bash
OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo/different_batchsize/Adamsemi_train__lr_1 --run_semi True  --model_path runs/demo/pretrain_lr_0.001/best.pth --semi_train__lr=0.0001 --batch_size=1 &
OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=1 python train_onemodel.py --save_dir demo/different_batchsize/Adamsemi_train__lr_8 --run_semi True  --model_path runs/demo/pretrain_lr_0.001/best.pth --semi_train__lr=0.0001 --batch_size=8 &
OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo/different_batchsize/Adamsemi_train__lr_16 --run_semi True  --model_path runs/demo/pretrain_lr_0.001/best.pth --semi_train__lr=0.000 --batch_size=16