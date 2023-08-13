export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline
#export WANDB_MODE=offline

# conda init

# # conda activate alpaca

# __conda_setup="$('/home/alex/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/home/alex/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "/home/alex/miniconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/home/alex/miniconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup

# conda activate alpaca

# echo $CUDA_VISIBLE_DEVICES

# nvidia-smi

#/home/alex/v3-score/llama_data_v2.json \
    #--model_name_or_path "/home/minghan/stanford_alpaca/alpaca_ckpt" \
# torchrun --nproc_per_node=4 --master_port=6006 train.py \
#     --model_name_or_path "/mnt/tjena/alex/llama/Llama-2-7b-hf" \
#     --data_path /home/alex/v3-score/llama_data_v3.json \
#     --has_instruction False \
#     --bf16 True \
#     --output_dir /mnt/tjena/alex/llama2_base_no_instr_7_26 \ # actually, this has instr tuning, but not rlhf
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 32 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --fsdp "full_shard auto_wrap" \
#     --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#     --tf32 True > finetuning_logs_7_26.out \

#--model_name_or_path "/mnt/tjena/LLaMA/LLaMA-hf-7B/-7b" \ 

# torchrun --nproc_per_node=1 --master_port=6007 
python train.py \
    --model_name_or_path "/mnt/tjena/alex/llama/Llama-2-7b-hf" \
    --data_path /home/alex/v3-score/llama_data_v4.json \
    --has_instruction True \
    --dataset_type="skg" \
    --bf16 True \
    --output_dir /mnt/tjena/alex/llama2_chat_7_31 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True #> finetuning_logs_7_31.out \
