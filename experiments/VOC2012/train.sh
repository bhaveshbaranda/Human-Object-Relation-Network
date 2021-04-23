#!/usr/bin/env bash

work_path=$(dirname $0)
cd ./${work_path}

python ./train_voca.py --network resnet50_v1d --dataset voca --gpus 0 --epochs 25 \
       --resume ./horelation_resnet50_v1d_voca_0014_nan.params --start-epoch 16 --max-lr 3e-5 --min-lr 1e-6 --cycle-len 30000 --seed 233 --verbose
