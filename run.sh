#!/bin/bash
device=$1
alg=$2

for (( s=0; s<=2; s++ ));
do
CUDA_VISIBLE_DEVICES=$device python -u $alg.py --seed=$s > $alg.log.$s
done
