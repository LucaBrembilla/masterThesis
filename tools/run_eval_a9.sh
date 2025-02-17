#!/bin/bash
PRED=/home/brembilla/exp/private_datasets/providentia/_predictions
GT=/home/brembilla/exp/private_datasets/providentia/_labels

PRED=/home/brembilla/exp/private_datasets/providentia/_predictions_less_computation
GT=/home/brembilla/exp/private_datasets/providentia/_labels

python eval_a9.py --pred ${PRED} \
 --gt ${GT} \
 --iou 0.1