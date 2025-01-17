#!/bin/bash

#SBATCH --job-name rnn_AT
#SBATCH --partition compute
#SBATCH --nodes 1
#SBATCH --mem 16GB
#SBATCH --account=COSC027924
#SBATCH -o ./outputs/AT/log_rnn.out
#SBATCH -e ./outputs/AT/log_rnn.err

# EDIT
export BP="AT"

echo "##### - Starting - #####"

# Load tensorflow
module load lang/python/anaconda/3.9.7-2021.12-tensorflow.2.7.0
module load lang/cuda/11.2-cudnn-8.1

# Load libcudart.so.11.0
export LD_LIBRARY_PATH=:$LD_LIBRARY_PATH:/sw/lang/cuda_11.2.2/targets/x86_64-linux/lib

# allocate async memory or smth (apparently it helps)
export TF_GPU_ALLOCATOR=cuda_malloc_async

python3 2_bit_model_tf.py ../../data/bp_models/${BP}/train.bin ../../data/bp_models/${BP}/541.LEELA.bin ../../data/bp_models/${BP}/525.X264.bin ../../data/bp_models/${BP}/502.GCC.bin ../../data/bp_models/${BP}/505.MCF.bin .outputs/${BP}/RNN_SAVED_MODEL