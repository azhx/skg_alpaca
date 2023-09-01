#!/bin/bash

#SBATCH --job-name=codellama_finetune
#SBATCH --mem=200G
#SBATCH -c 10
#SBATCH --partition=a100
#SBATCH --qos=a100_wenhuchen
#SBATCH -w gpu181
#SBATCH --output=%x.%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:4

export WANDB_DISABLED=True

#MODEL_DIR=/ssd005/projects/waterloo_nlp/alex/Llama-2-7b-hf/
#MODEL_DIR=/ssd005/projects/waterloo_nlp/alex/llama2_8_18/before_restart/checkpoint-15380
#MODEL_DIR=/h/wenhuchen/STORAGE/alex/codellama/CodeLlama-7b-Instruct-hf
MODEL_DIR=/h/wenhuchen/STORAGE/alex/codellama_finetune/checkpoint-25630
DATA_PATH=/ssd005/projects/waterloo_nlp/alex/llama_data_v4.json
OUTPUT_DIR=/ssd005/projects/waterloo_nlp/alex/codellama_finetune/9_1

#LAUNCHER="python"
LAUNCHER="deepspeed"
SCRIPT="train.py"
SCRIPT_ARGS=(--model_name_or_path ${MODEL_DIR} \
--data_path "${DATA_PATH}" \
--output_dir ${OUTPUT_DIR} \
--pkl_path "codellama_dataset.pkl" \
--has_instruction True \
--dataset_type="skg" \
--bf16 True \
--num_train_epochs 5 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 16 \
--evaluation_strategy no \
--save_strategy "epoch" \
--learning_rate 2e-6 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--model_max_length 1024 \
--logging_steps 1 \
--tf32 True \
--deepspeed configs/default_offload_opt_param.json
)

echo 'address:'${MASTER_ADDR},'SURM_JOBID:'${SLURM_PROCID}
echo $LAUNCHER $SCRIPT "${SCRIPT_ARGS[@]}"

$LAUNCHER $SCRIPT "${SCRIPT_ARGS[@]}"
