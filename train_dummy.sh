#!/bin/bash
mkdir -p checkpoints
python -u train.py --name raft-sintel --stage sintel --validation sintel --gpus 0 --num_steps 101 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85
