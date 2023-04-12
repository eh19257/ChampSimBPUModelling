#!/bin/bash

#SBATCH --job-name bp_model_2_bit_tf_transformer_pwd
#SBATCH --partition gpu
#SBATCH --nodes 2
#SBATCH --gres gpu:2
#SBATCH --mem 48GB
#SBATCH --account=COSC027924
#SBATCH -o ./outputs/log_%j.out
#SBATCH -e ./outputs/log_%j.err


echo "##### - Starting - #####"

# Load tensorflow
module load lang/python/anaconda/3.9.7-2021.12-tensorflow.2.7.0
module load lang/cuda/11.2-cudnn-8.1

# Load libcudart.so.11.0
export LD_LIBRARY_PATH=:$LD_LIBRARY_PATH:/sw/lang/cuda_11.2.2/targets/x86_64-linux/lib

python3 2_bit_model_tf_transformer.py ../../data/bp_models/2_bit/pwd.bin ../../data/bp_models/2_bit/ls.bin SAVED_MODEL_tf_transformer.bin