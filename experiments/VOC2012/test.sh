#!/usr/bin/env bash
#/kaggle/working/horelation_resnet50_v1d_voca_0014_nan.params
work_path=$(dirname $0)
cd ./${work_path}

python ./eval_voca.py --network resnet50_v1d --dataset voca --gpus 0 \
 --/kaggle/working/horelation_resnet50_v1d_voca_0014_nan.params --save-outputs
