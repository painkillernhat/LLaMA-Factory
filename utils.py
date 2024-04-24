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

def jsonl_to_json(jsonl_file_path, output_json_file_path):
    """convert jsonl to json

    Args:
        jsonl_file_path (_type_): _description_
        output_json_file_path (_type_): _description_
    """
    with open(jsonl_file_path, 'r') as jsonl_file:
        lines = jsonl_file.readlines()

    json_data = [json.loads(line.strip()) for line in lines]
    with open(output_json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

def add_new_dataset_info(dataset_info_path, name, path):
    """add new dataset info to dataset_info.json

    Args:
        dataset_info_path (_type_): _description_
        name (_type_): _description_
        path (_type_): _description_
    """
    with open(dataset_info_path, 'r') as file:
        data = json.load(file)

    if name in data:
        del data[name]

    template = data['template']
    data[name] = copy.deepcopy(template)
    data[name]['file_name'] = path

    with open(dataset_info_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

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
        
def load_json(file_path):
    """Loads JSON data from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    """Saves JSON data to a file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def expand_dataset(args, training_data_path, n, k, output_file_path, num_repeat):
    """Expands the dataset by running inference on n random entries from the training dataset, repeated k times,
    updating history after obtaining outputs from just the previous record."""
    training_data = load_json(training_data_path)
    expanded_data = list(training_data)  # Start with the original dataset

    client = OpenAI(
        base_url=f"http://localhost:{args.api_port}/v1",
        api_key="token-abc123"
    )

    for iteration in range(k):
        # Only select entries if needed, otherwise use the last entry repeatedly
        if len(training_data) > 0:
            selected_entries = random.sample(training_data, min(n, len(training_data)))
        else:
            selected_entries = [expanded_data[-1]] * n  # Use the last entry if no more new data

        for _ in range(num_repeat):
            last_entry = None
            for sample in selected_entries:
                # Prepare the sample with history from the last entry if available
                sample_with_history = {
                    "instruction": sample['instruction'],
                    "input": "",
                    "history": [[last_entry['instruction'], last_entry['output']]] if last_entry else []
                }

                # Perform inference to get the output
                response = perform_inference(client, args, args.adapter_path, sample_with_history)
                sample_with_history['output'] = response
                expanded_data.append(sample_with_history)

                last_entry = sample_with_history

        save_json(expanded_data, output_file_path)
        print(f"Dataset expanded and inferred, updated and saved to {output_file_path}")

    return expanded_data

def expand_dataset(args, training_data_path, n, k, output_dir, num_repeat):
    """Expands the dataset by running inference on n random entries from the training dataset, repeated k times,
    saving each batch for inference, and updating history sequentially."""
    training_data = load_json(training_data_path)
    expanded_data = list(training_data)  # Start with the original dataset

    # Create the directory for inference files if it does not exist
    inference_dir = os.path.join(output_dir, "inference_batches")
    os.makedirs(inference_dir, exist_ok=True)

    client = OpenAI(
        base_url=f"http://localhost:{args.api_port}/v1",
        api_key="token-abc123"
    )

    for iteration in range(k):
        selected_entries = random.sample(training_data, min(n, len(training_data)))

        for batch_index in range(num_repeat):
            batch_data = []
            last_entry = None

            for sample in selected_entries:
                # Prepare the sample with history from the last entry if available
                sample_with_history = {
                    "instruction": sample['instruction'],
                    "input": "",
                    "history": [[last_entry['instruction'], last_entry['output']]] if last_entry else []
                }
                batch_data.append(sample_with_history)
                last_entry = sample  # Update last_entry for the next sample's history

            # Save the current batch to a JSON file
            batch_file_path = os.path.join(inference_dir, f"batch_{iteration}_{batch_index}.json")
            save_json(batch_data, batch_file_path)

            # Perform inference on the saved batch file
            inferred_data = perform_inference(client, args, args.adapter_path, batch_file_path)

            # Update the main dataset with the inferred outputs and save the dataset
            for original, inferred in zip(batch_data, inferred_data):
                original['output'] = inferred['output']
                expanded_data.append(original)

    # Save the final expanded dataset
    final_output_path = os.path.join(output_dir, "final_dataset.json")
    save_json(expanded_data, final_output_path)
    print(f"Final expanded dataset saved to {final_output_path}")

    return expanded_data
    
def perform_inference(client, args, sft_full_path, testset):
    if args.is_using_vllm:
        template = "llama2" if "llama" in args.model_name_or_path.lower() else "mistral"

        deploy_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} API_PORT={args.api_port} python src/api_demo.py \
            --model_name_or_path {sft_full_path} \
            --template {template} \
            --infer_backend vllm \
            --vllm_enforce_eager"""
        
        print("Deploying LLM...")
        server_process = run_server(deploy_command)
        time.sleep(60)

        # Inference
        # client = OpenAI(base_url=f"http://localhost:{args.api_port}/v1", api_key="token-abc123")
        # testset_path = f"{args.dataset_dir}/{testset}.json"
        test_data = load_json(testset)

        predictions = []
        for sample in tqdm(test_data):
            completion = client.chat.completions.create(
                model=sft_full_path,
                messages=[{"role": "user", "content": sample['instruction']}]
            )
            sample['output'] = completion.choices[0].message.content
            predictions.append(sample)

        output_file_path = f"{args.dataset_dir}/generated_predictions.json"
        save_json(predictions, output_file_path)
        print(f"Predictions saved to: {output_file_path}")

        # Shutdown server
        shutdown_server(f"kill {server_process.pid}")
    else:
        generate_text_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/train_bash.py \
            --stage sft --do_predict --model_name_or_path {sft_full_path} --dataset {testset} \
            --dataset_dir {args.dataset_dir} --template {args.template} --finetuning_type {args.finetuning_type} \
            --output_dir {sft_full_path} --cutoff_len {args.cutoff_len} \
            --per_device_eval_batch_size {args.per_device_eval_batch_size} --predict_with_generate --report_to none --fp16"""
        run_cli_command(generate_text_command)