import numpy as np
import json
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import argparse

login("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")

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def parse_data(prompted_record):
    """ Parses the prompted records to extract question and answer pairs. """
    return [{'question': entry['input'], 'answer': entry['output']} for entry in prompted_record]

def compute_rewards(data, model, tokenizer, num_questions, num_mc_samples, num_users):
    max_new_tokens = 20  # Set how many new tokens to generate
    
    contexts = []

    for user_index in range(num_users):
        for question_index in range(len(data)):
            current_qa = data[:question_index + 1]
            context = " ".join([f"{qa['question']} {qa['answer']}" for qa in current_qa])
            inputs = tokenizer(context, return_tensors="pt")
            try:
                for q_index in range(num_questions):
                    for mc_index in range(num_mc_samples):
                        output = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
                        outcome = tokenizer.decode(output[0], skip_special_tokens=True)
                        contexts.append((context, outcome))
            except Exception as e:
                print(f"Error during generation for context: {context}\nError: {str(e)}")

    rewards = {}
    for context, outcome in contexts:
        if context not in rewards:
            rewards[context] = []
        rewards[context].append(outcome)
    
    average_rewards = {ctx: np.mean(outs) for ctx, outs in rewards.items()}
    return average_rewards

def save_results_to_json(results, output_file):
    """Saves the computed results to a JSON file."""
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="")
    args = parser.parse_args()
    
    dataset = read_json("data/animal_guessing/animal_guessing_prompt.json")
    model_id = "meta-llama/llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    num_questions = len(dataset[0]['original_record']['history'])
    num_mc_samples = 2
    num_users = len(dataset)
    
    for index, record in enumerate(dataset):
        print("-------")
        prompted_record = record["prompted_record"]
        data = parse_data(prompted_record)
        rewards = compute_rewards(data, model, tokenizer, num_questions, num_mc_samples, num_users)
        save_results_to_json(rewards, f"data/animal_guessing/reward/animal_guessing_{index}.json")