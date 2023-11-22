# We include a simple full-parameter finetuning & inference script here. Our V0.1 chat model is finetuned using this script. 
# The FT dataset we use is openassistant-guanaco. For finetuning with less than 4GB RAM, we refer you to the Qlora and bitsandbytes repo.
# We did not undergone extensive hyperparameter tuning nor choosing more performant FT datasets. 
# We hope the community can explore on finetuning TinyLlama and come up with better chat models. I will include community-finetuned models in this repo.

# V0.1
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --multi_gpu --num_processes 4 --main_process_port 1234 finetune.py \
    --model_name_or_path PY007/TinyLlama-1.1B-intermediate-step-240k-503b \
    --output_dir ./output/503B_FT_lr1e-5_ep5 \
    --logging_steps 10 \
    --save_strategy epoch \
    --data_seed 42 \
    --save_total_limit 6 \
    --evaluation_strategy epoch \
    --eval_dataset_size 512 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length=False \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --warmup_ratio 0.05 \
    --lr_scheduler_type constant \
    --dataset oasst1 \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 4 \
    --max_steps 0 \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --seed 0 \
    --trust_remote_code \
    --report_to wandb 


# V0.2
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 --main_process_port 1234 finetune.py \
    --model_name_or_path PY007/TinyLlama-1.1B-intermediate-step-480k-1T \
    --output_dir ./output/503B_FT_lr1e-5_ep5_top1_2023-08-25 \
    --logging_steps 10 \
    --save_strategy epoch \
    --data_seed 42 \
    --save_total_limit 6 \
    --evaluation_strategy epoch \
    --eval_dataset_size 512 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length=False \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --warmup_ratio 0.05 \
    --lr_scheduler_type constant \
    --dataset OpenAssistant/oasst_top1_2023-08-25 \
    --dataset_format oasst1 \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 4 \
    --max_steps 0 \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --seed 0 \
    --trust_remote_code \
    --report_to wandb 
