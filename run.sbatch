#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=0:30:00
#SBATCH --job-name="nlu_hw4"
#SBATCH --output=test_finetune.out
module purge
module load cuda/10.0.130
module load gcc/6.3.0

# set environment variable
NETID=wx455
#conda activate /scratch/${NETID}/nlu/env
export PYTORCH_TRANSFORMERS_CACHE=/scratch/${NETID}/nlu/cache
TASK_NAME=boolq
DATA_DIR=/scratch/${NETID}/nlu/data
export PYTHONPATH=/scratch/${NETID}/nlu/code/transformers/src:$PYTHONPATH


cd /scratch/${NETID}/nlu/code/transformers

# run python
python ./examples/run_glue.py \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir ${DATA_DIR}/${TASK_NAME} \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /scratch/${NETID}/nlu/${TASK_NAME}_run/
