#!/bin/bash

#SBATCH --job-name bp_model_2_bit_tf_transformer_pwd
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --mem 32GB
#SBATCH --account=COSC027924
#SBATCH -o ./outputs/log_%j.out
#SBATCH -e ./outputs/log_%j.err


cd "${SLURM_SUBMIT_DIR}"

echo "${SLURM_SUBMIT_DIR}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${SLURM_JOBID}"
echo This jobs runs on the following machines:
echo "${SLURM_JOB_NODELIST}"

python3 2_bit_model_tf_transformer.py ../../data/bp_models/2_bit/pwd.bin ../../data/bp_models/2_bit/pwd.bin "SAVED_MODEL_tf_transformer"