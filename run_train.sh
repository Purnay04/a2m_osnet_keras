#!/bin/bash

set -e

usage()
{
    echo
    echo "Usage: ./trash_new.sh <model_name>"
    echo
    echo "where <model_name> could be one of the following:"
    echo " osnet"
    echo
}

if [ $# -ne 1 ]; then
    usage
    exit
fi

case $1 in
    osnet )
        CUDA_VISIBLE_DEVICES=1 python train.py --dropout_rate 0.2 \
                         --optimizer adam --batch_size 16 \
                         --lr_sched linear --initial_lr 1e-3 --final_lr 1e-4 \
                         --epochs 50 osnet
        ;;
    * )
        usage
        exit
esac