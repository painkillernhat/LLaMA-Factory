from src.llmtuner import run_exp

import argparse
import json
import random
import os
import copy
import math

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
    train_dataset = os.path.join(dataset_dir, "/animal_guessing/splits/animal_guessing_train.json")
    test_dataset = os.path.join(dataset_dir, "/animal_guessing/splits/animal_guessing_test.json")
    
    model_name = "llama-2-7b"
    output_dir = os.path.join("saves", model_name, "animal_guessing", "supervised")
    
    # Fine-tune the model on the training set
    ft_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port} \
        --config_file examples/accelerate/default.yaml \
        src/train_bash.py \
        --do_train \
        --do_eval \
        --model_name_or_path {model_name} \
        --dataset {train_dataset} \
        --output_dir {output_dir} \
        --overwrite_cache \
        --overwrite_output_dir \
        --per_device_train_batch_size {args.per_device_train_batch_size} \
        --per_device_eval_batch_size {args.per_device_eval_batch_size} \
        --gradient_accumulation_steps {args.gradient_accumulation_steps} \
        --learning_rate {args.learning_rate} \
        --num_train_epochs {args.num_train_epochs} \
        --max_samples {args.max_samples} \
        --val_size {args.val_size} \
        --plot_loss
        """
    
    print("Fine-tuning the model...")
    run_cli_command(ft_command)
    
    # Evaluate the fine-tuned model on the test set
    eval_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port} \
        --config_file examples/accelerate/default.yaml \
        src/train_bash.py \
        --do_eval \
        --model_name_or_path {output_dir} \
        --dataset {test_dataset} \
        --output_dir {output_dir} \
        --per_device_eval_batch_size {args.per_device_eval_batch_size}
        """
    
    print("Evaluating the finetuned model...")
    run_cli_command(eval_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--main_process_port", type=int, default=12345)
    parser.add_argument("--dataset_dir", type=str, default="data")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--val_size", type=float, default=0.1)
    args = parser.parse_args()

    main(args)