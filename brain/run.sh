#! /bin/bash

# ================================ SPLIT = (0.7, 0.15, 0.15) ================================
python3 train.py --epochs 50 --model-output ../new_models/model_00 --filters 32 --num-blocks 4 --batch-size 16
python3 train.py --epochs 50 --model-output ../new_models/model_01 --filters 32 --num-blocks 6 --batch-size 16
python3 train.py --epochs 50 --model-output ../new_models/model_02 --filters 32 --num-blocks 8 --batch-size 16

python3 train.py --epochs 50 --model-output ../new_models/model_03 --filters 32 --num-blocks 4 --batch-size 32
python3 train.py --epochs 50 --model-output ../new_models/model_04 --filters 32 --num-blocks 6 --batch-size 32
python3 train.py --epochs 50 --model-output ../new_models/model_05 --filters 32 --num-blocks 8 --batch-size 32

python3 train.py --epochs 50 --model-output ../new_models/model_06 --filters 32 --num-blocks 4 --batch-size 16 --resize-shape 128 128
python3 train.py --epochs 50 --model-output ../new_models/model_07 --filters 32 --num-blocks 6 --batch-size 16 --resize-shape 128 128
python3 train.py --epochs 50 --model-output ../new_models/model_08 --filters 32 --num-blocks 8 --batch-size 16 --resize-shape 128 128

python3 train.py --epochs 50 --model-output ../new_models/model_09 --filters 32 --num-blocks 4 --batch-size 32 --resize-shape 128 128
python3 train.py --epochs 50 --model-output ../new_models/model_10 --filters 32 --num-blocks 6 --batch-size 32 --resize-shape 128 128
python3 train.py --epochs 50 --model-output ../new_models/model_11 --filters 32 --num-blocks 8 --batch-size 32 --resize-shape 128 128

# ================================ SPLIT = (0.6, 0.2, 0.2) ================================
python3 train.py --epochs 50 --model-output ../new_models/model_12 --filters 32 --num-blocks 4 --batch-size 16 --split 0.6 0.2 0.2
python3 train.py --epochs 50 --model-output ../new_models/model_13 --filters 32 --num-blocks 6 --batch-size 16 --split 0.6 0.2 0.2
python3 train.py --epochs 50 --model-output ../new_models/model_14 --filters 32 --num-blocks 8 --batch-size 16 --split 0.6 0.2 0.2

python3 train.py --epochs 50 --model-output ../new_models/model_15 --filters 32 --num-blocks 4 --batch-size 32 --split 0.6 0.2 0.2
python3 train.py --epochs 50 --model-output ../new_models/model_16 --filters 32 --num-blocks 6 --batch-size 32 --split 0.6 0.2 0.2
python3 train.py --epochs 50 --model-output ../new_models/model_17 --filters 32 --num-blocks 8 --batch-size 32 --split 0.6 0.2 0.2

python3 train.py --epochs 50 --model-output ../new_models/model_18 --filters 32 --num-blocks 4 --batch-size 16 --split 0.6 0.2 0.2 --resize-shape 128 128
python3 train.py --epochs 50 --model-output ../new_models/model_19 --filters 32 --num-blocks 6 --batch-size 16 --split 0.6 0.2 0.2 --resize-shape 128 128
python3 train.py --epochs 50 --model-output ../new_models/model_20 --filters 32 --num-blocks 8 --batch-size 16 --split 0.6 0.2 0.2 --resize-shape 128 128

python3 train.py --epochs 50 --model-output ../new_models/model_21 --filters 32 --num-blocks 4 --batch-size 32 --split 0.6 0.2 0.2 --resize-shape 128 128
python3 train.py --epochs 50 --model-output ../new_models/model_22 --filters 32 --num-blocks 6 --batch-size 32 --split 0.6 0.2 0.2 --resize-shape 128 128
python3 train.py --epochs 50 --model-output ../new_models/model_23 --filters 32 --num-blocks 8 --batch-size 32 --split 0.6 0.2 0.2 --resize-shape 128 128