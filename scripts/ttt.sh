#!/bin/bash

# custom config
DATASET = $1
SEED = $2
TRHESH = $3
BS_TRAIN = $4
BS_TEST = $5

CUDA_VISIBLE_DEVICES=0 python main_ttt.py \
--dataset_config ./configs/datasets/${DATASET}.yaml \
--seed ${SEED} \
--encoder both \
--position all \
--backbone "ViT-B/16" \
--lora_r 2 \
--train_epoch 20 \
--alpha 0.05 \
--ratio 0.95 \
--thresh ${TRHESH} \
--ttt 1 \
--reg 1 \
--consistency 1 \
--learning_rate_base 5e-4 \
--learning_rate_ttt 5e-3 \
--train_batch_size ${BS_TRAIN} \
--eval_batch_size ${BS_TEST}
