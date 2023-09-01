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

# #python train.py \
# torchrun --nproc_per_node=1 --master_port=6007 train.py \
#     --model_name_or_path "/mnt/tjena/alex/llama/Llama-2-7b-hf" \
#     --data_path /home/alex/v3-score/llama_data_v4.json \
#     --has_instruction True \
#     --dataset_type="skg" \
#     --bf16 True \
#     --output_dir /mnt/tjena/alex/llama2_chat_7_31 \
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
#     --tf32 True #> finetuning_logs_7_31.out \
#     # --fsdp "full_shard auto_wrap" \
#     # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \

    #--model_name_or_path "/mnt/tjena/alex/llama/Llama-2-7b-hf" \

#torchrun --nproc_per_node=1 --master_port=6007 train.py \
python train.py \
    --model_name_or_path "/h/wenhuchen/STORAGE/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9"\
    --data_path /h/wenhuchen/STORAGE/alex/llama_data_v4.json \
    --has_instruction True \
    --dataset_type="skg" \
    --bf16 True \
    --output_dir /h/wenhuchen/STORAGE/alex/llama2_8_18 \
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
    --model_max_length 1024 \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --tf32 True
    # > /h/wenhuchen/STORAGE/alex/finetuning_logs_7_31.out \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
