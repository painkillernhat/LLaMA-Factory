from src.llmtuner import run_exp
import argparse
import json
import random
import os
import subprocess
import time
import copy
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from huggingface_hub import login

########## define variables ##########

questions = {
    "Class_Type": "What is the animal's class type?",
    "hair": "Does the animal have hair?",
    "feathers": "Does the animal have feathers?",
    "eggs": "Does the animal lay eggs?",
    "milk": "Does the animal produce milk?",
    "airborne": "Is the animal airborne?",
    "aquatic": "Is the animal aquatic?",
    "predator": "Is the animal a predator?",
    "toothed": "Does the animal have teeth?",
    "backbone": "Does the animal have a backbone?",
    "breathes": "Does the animal breathe air?",
    "venomous": "Is the animal venomous?",
    "fins": "Does the animal have fins?",
    "legs": "How many legs does the animal have?",
    "tail": "Does the animal have a tail?",
    "domestic": "Is the animal domesticated?",
    "catsize": "Is the animal at catsize?"
}

########## functions ##########

def run_cli_command(command):
    """run cli command

    Args:
        command (_type_): _description_
    """
    os.system(command)


def run_server(cmd_string):
    """running the server

    Args:
        cmd_string (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        server_process = subprocess.Popen(cmd_string, shell=True)
        return server_process
    except Exception as e:
        print(f"Error starting server: {e}")
        return None

def shutdown_server(process):
    """shutting down the server

    Args:
        process (_type_): _description_
    """
    try:
        process.terminate()
        print("Server shutdown successfully.")
    except Exception as e:
        print(f"Error shutting down server: {e}")

def find_and_kill_process(command):
    """ kill the process """
    find_pid_command = f"""pgrep -af "{command}" """
    pid_output = subprocess.check_output(find_pid_command, shell=True)
    pid_lines = pid_output.decode().splitlines()
    pids = [line.split()[0] for line in pid_lines]

    print("PID(s) of the process:")
    print(pids)

    if pids:
        kill_pid_command = f"kill -9 {' '.join(pids)}"
        subprocess.run(kill_pid_command, shell=True)
        print("Process(es) killed.")
    else:
        print("No matching process found.")

def generate_initial_response(question, value):
    if question == "What is the animal's class type?":
        return f"It is a {value}."
    elif question == "How many legs does the animal have?":
        return f"It has {value} legs."
    elif question == "Does the animal have hair?":
        return "Yes, it has hair." if value == 1 else "No, it doesn't have hair."
    elif question == "Does the animal have feathers?":
        return "Yes, it has lots of feathers." if value == 1 else "No, it doesn't have feathers."
    elif question == "Does the animal lay eggs?":
        return "Yes, it lays eggs." if value == 1 else "No, it doesn't lay eggs."
    elif question == "Does the animal produce milk?":
        return "Yes, it produces milk." if value == 1 else "No, it doesn't produce milk."
    elif question == "Is the animal airborne?":
        return "Yes, it is airborne." if value == 1 else "No, it isn't airborne."
    elif question == "Is the animal a predator?":
        return "Yes, it is a predator." if value == 1 else "No, it isn't a predator."
    elif question == "Does the animal have teeth?":
        return "Yes, it has teeth." if value == 1 else "No, it doesn't have teeth."
    elif question == "Does the animal have a backbone?":
        return "Yes, it has a backbone." if value == 1 else "No, it doesn't have a backbone."
    elif question == "Is the animal venomous?":
        return "Yes, it is venomous." if value == 1 else "No, it isn't venomous."
    elif question == "Does the animal have fins?":
        return "Yes, it has fins." if value == 1 else "No, it doesn't have fins."
    elif question == "Does the animal have a tail?":
        return "Yes, it has a tail." if value == 1 else "No, it doesn't have a tail."
    elif question == "Is the animal domesticated?":
        return "Yes, it is domesticated." if value == 1 else "No, it isn't domesticated."
    elif question == "Is the animal at catsize?":
        return "Yes, it is at catsize." if value == 1 else "No, it isn't at catsize."
    elif question == "Is the animal aquatic?":
        return "Yes, it is aquatic." if value == 1 else "No, it isn't aquatic."
    elif question == "Does the animal breathe air?":
        return "Yes, it breathes air." if value == 1 else "No, it doesn't breathe air."
    else:
        return "Yes." if value == 1 else "No."
    
def init_dataset(csv_file, file_path):
    data = pd.read_csv(csv_file, sep=';')
    all_data = []

    for index, row in data.iterrows():
        for col, question in questions.items():
            answer = generate_initial_response(question, row[col])
            item = {
                "instruction": question,
                "input": "",
                "output": answer,
                "history": []
            }
            all_data.append(item)

    with open(file_path, 'w') as outfile:
        json.dump(all_data, outfile, indent=4)
    return all_data

def eliminate_outcome(old_file, new_file):
    with open(old_file, 'r') as f:
        data = json.load(f)

    filtered_data = [entry for entry in data if entry['instruction'] not in ["I will adopt the animal.", "I will not adopt the animal."]]
    with open(new_file, 'w') as file:
        json.dump(filtered_data, file, indent=4)
    print("Data has been updated and saved.")
            
def load_json(file_path):
    """Loads JSON data from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    """Saves JSON data to a file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def expand_dataset(training_data_path, n, k, num_repeat, output_file_path):
    original_data = load_json(training_data_path)
    all_data = []

    client = None

    histories = [[] for _ in range(n * num_repeat)]
    for iteration in range(1, k + 1):
        batch_samples = []
        for p in range(n):
            selected_entry = random.sample(original_data, num_repeat)
            for i, entry in enumerate(selected_entry):
                new_entry = {
                    "instruction": entry['instruction'],
                    "input": "",
                    "history": histories[p * num_repeat + i].copy()
                }
                histories[p * num_repeat + i].append([entry['instruction'], entry['output']])
                batch_samples.append(new_entry)
        
        inference_path = 'data/animal/inference'
        if os.path.exists(inference_path) is False:
            os.mkdir(inference_path)
        
        batch_file_path = f"{inference_path}/batch_data_iteration_{iteration}.json"
        save_json(batch_samples, batch_file_path)
        print(f"Batch data for iteration {iteration} saved to {batch_file_path}")

        # responses = perform_inference(client, adapter_path, batch_samples)
        # for i, response in enumerate(responses):
        #     batch_samples[i]['output'] = response
        
        all_data.extend(batch_samples)
        save_json(all_data, output_file_path)
        print(f"Iteration {iteration} completed and saved to {output_file_path}")

    return all_data
    
# def perform_inference(args, client, sft_full_path, testset):
#     if args.is_using_vllm:
#         template = "llama2" if "llama" in args.model_name_or_path.lower() else "mistral"

#         deploy_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} API_PORT={args.api_port} python src/api_demo.py \
#             --model_name_or_path {sft_full_path} \
#             --template {template} \
#             --infer_backend vllm \
#             --vllm_enforce_eager"""
        
#         print("Deploying LLM...")
#         server_process = run_server(deploy_command)
#         time.sleep(60)

#         # Inference
#         # client = OpenAI(base_url=f"http://localhost:{args.api_port}/v1", api_key="token-abc123")
#         # testset_path = f"{args.dataset_dir}/{}/{testset}.json"
#         test_data = load_json(testset)

#         predictions = []
#         for sample in tqdm(test_data):
#             completion = client.chat.completions.create(
#                 model=sft_full_path,
#                 messages=[{"role": "user", "content": sample['instruction']}]
#             )
#             sample['output'] = completion.choices[0].message.content
#             predictions.append(sample)

#         output_file_path = f"{args.dataset_dir}/generated_predictions.json"
#         save_json(predictions, output_file_path)
#         print(f"Predictions saved to: {output_file_path}")

#         # Shutdown server
#         shutdown_server(f"kill {server_process.pid}")
#     else:
#         generate_text_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
#             --stage sft --do_predict --model_name_or_path {sft_full_path} --dataset {testset} \
#             --dataset_dir {args.dataset_dir} --template {args.template} --finetuning_type {args.finetuning_type} \
#             --output_dir {sft_full_path} --cutoff_len {args.cutoff_len} \
#             --per_device_eval_batch_size {args.per_device_eval_batch_size} --predict_with_generate --report_to none --fp16"""
#         run_cli_command(generate_text_command)