from src.llmtuner import run_exp
import numpy as np
import argparse
import json
import random
import os
import copy
import math
import os
import subprocess

os.environ['MKL_THREADING_LAYER'] = 'GNU'

from huggingface_hub import login

login("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")

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def count_dataset_len(file_path):
    data = read_json(file_path)
    return len(data)

def run_cli_command(command):
    os.system(command)

def main(args):
    # Prepare dataset
    dataset_dir = args.dataset_dir
    train_dataset = "animal_guessing/animal_guessing_prompt.json"
    
    # test_dataset = "animal_guessing_test.json"

    # sft_train_command = f"""python src/train_bash.py --stage sft --template {args.template} --model_name_or_path {args.model_name_or_path} --dataset animal_guessing_train --num_train_epochs 1 --output_dir {args.output_dir}"""
    
    sft_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port} \
        src/train_bash.py \
        --stage sft \
        --do_train \
        --do_eval \
        --use_fast_tokenizer True \
        --template {args.template} \
        --model_name_or_path {args.model_name_or_path} \
        --dataset animal_guessing_test \
        --output_dir {args.output_dir} \
        --overwrite_cache \
        --overwrite_output_dir \
        --bf16 True \
        --tf32 False \
        --learning_rate {args.learning_rate} \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --weight_decay 0.001 \
        --warmup_ratio 0.02 \
        --per_device_train_batch_size {args.per_device_train_batch_size} \
        --per_device_eval_batch_size {args.per_device_eval_batch_size} \
        --gradient_accumulation_steps {args.gradient_accumulation_steps} \
        --num_train_epochs {args.num_train_epochs} \
        --max_samples {args.max_samples} \
        --val_size {args.val_size} \
        --save_steps {args.save_steps} \
        --logging_steps {args.logging_steps} \
        --neftune_noise_alpha 0 \
        --plot_loss True \
        --report_to none
        """
    
    print("Supervised finetuning the model...")
    run_cli_command(sft_command)
    # process = subprocess.run(sft_train_command, shell=True, text=True, capture_output=True)
    
    # # Print the outputs and any errors
    # print("STDOUT:", process.stdout)
    # print("STDERR:", process.stderr)

    print("----------------------------------")

    # ppo_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port} \
    #     src/train_bash.py \
    #     --stage ppo \
    #     --do_train \
    #     --template {args.template} \
    #     --model_name_or_path {args.model_name_or_path} \
    #     --reward_model {args.reward_model} \
    #     --dataset animal_guessing_train \
    #     --output_dir {args.output_dir}_ppo \
    #     --num_train_epochs {args.ppo_num_train_epochs} \
    #     --per_device_train_batch_size {args.ppo_per_device_train_batch_size} \
    #     --learning_rate {args.ppo_learning_rate} \
    #     --seed {args.seed} \
    #     --report_to wandb
    #     """
    
    # print("PPO fine-tuning the model...")
    # run_cli_command(ppo_command)
    
    # Evaluate the fine-tuned model on the test set
    # eval_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port} \
    #     src/train_bash.py \
    #     --do_eval \
    # 	--template {args.template} \
    #     --model_name_or_path {args.model_name_or_path} \
    #     --dataset animal_guessing_test \
    #     --output_dir {args.output_dir} \
    #     --per_device_eval_batch_size {args.per_device_eval_batch_size}
    #     """
    
    # print("Evaluating the finetuned model...")
    # run_cli_command(eval_command)

if __name__ == "__main__":
    
    # dataset_path = os.path.join(args.dataset_dir, "animal_guessing_train.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=str, default="llama2", help="Template name")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="")
    parser.add_argument("--main_process_port", type=int, default=29500, help="Port")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Number of training epochs")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max samples")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="saves/meta-llama/Llama-2-7b-hf/animal_guessing")
    parser.add_argument("--data_info_path", type=str, default="data/dataset_info.json", help="Path to dataset info")
    parser.add_argument("--use_accelerate", type=bool, default=True, help="is using accelerate")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save steps")
    # parser.add_argument("--ppo_num_train_epochs", type=int, default=3, help="Number of training epochs for PPO")
    # parser.add_argument("--ppo_per_device_train_batch_size", type=int, default=8, help="Batch size for PPO training")
    # parser.add_argument("--ppo_learning_rate", type=float, default=1e-5, help="Learning rate for PPO")
    # parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # parser.add_argument("--reward_model", type=str, default="meta-llama/Llama-2-7b-hf", help="Path or name of the reward model")
    args = parser.parse_args()

    main(args)