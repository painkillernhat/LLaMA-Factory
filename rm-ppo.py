import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import json
import argparse

login("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")

def calculate_rewards(model_name, data):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    num_questions = len(data[0]['history']) - 1
    rewards = torch.zeros(num_questions)

    for user_data in data:
        history = user_data['history']
        
        for question_id in range(num_questions):
            context = ' '.join([f"{q} {a}" for q, a in history[:question_id+1]])
            
            outcome_sum = 0
            mc_samples = []
            
            treatment_context = f"{context} {history[question_id+1][0]}"
            outcome_str = generate_outcome(model, tokenizer, treatment_context)
            mc_samples.append(outcome_str)

            outcome_sum = sum(1 if "The outcome is 1" in sample else 0 for sample in mc_samples)
            average_outcome = outcome_sum / len(mc_samples)
            rewards[question_id] += average_outcome

    rewards /= len(data)
    return rewards

def generate_outcome(model, tokenizer, context):
    input_ids = tokenizer.encode(context, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    outcome_str = tokenizer.decode(output[0], skip_special_tokens=True)
    return outcome_str

def jsonl_to_json(jsonl_file_path, output_json_file_path):
    # Read JSONL file
    with open(jsonl_file_path, 'r') as jsonl_file:
        lines = jsonl_file.readlines()

    # Parse each line as JSON and store in a list
    json_data = [json.loads(line.strip()) for line in lines]

    # Write the list of JSON objects to a JSON file
    with open(output_json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

def main(args):
    # if args.dataset == "animal":
    #     dataset = "data/animal_guessing/animal_guessing.jsonl"
    #     new_dataset = "data/animal_guessing/animal_guessing.json"
    # else:
    #     print("nothing!")
        
    model_name = "meta-llama/Llama-2-7b-hf"
    # data = jsonl_to_json(dataset, new_dataset)
    data = "data/animal_guessing/animal_guessing.json"
    
    rewards = calculate_rewards(model_name, data)
    print("Rewards:", rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="animal")
    args = parser.parse_args()
    
    main(args)