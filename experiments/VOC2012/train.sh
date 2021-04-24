#!/usr/bin/env bash
#--resume /kaggle/working/horelation_resnet50_v1d_voca_0014_nan.params#
work_path=$(dirname $0)
cd ./${work_path}

python ./train_voca.py --network resnet50_v1d --dataset voca --gpus 0 --epochs 14 \
        --resume /kaggle/working/horelation_resnet50_v1d_voca_0012_0.8818.params --start-epoch 13 --max-lr 3e-5 --min-lr 1e-6 --cycle-len 30000 --seed 233 --verbose
