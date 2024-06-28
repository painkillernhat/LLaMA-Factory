import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from huggingface_hub import login
import json
import argparse
from llmtuner.train.ppo.trainer import PPOTrainer

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

def load_jsonl_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line.strip()) for line in file]
    return data

def main(args):
    # Load the dataset
    dataset_path = args.dataset_path
    data = load_jsonl_data(dataset_path)
    
    # Define the model and tokenizer
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Calculate the rewards
    rewards = calculate_rewards(model_name, data)
    print("Rewards:", rewards)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        seed=args.seed,
    )
    
    # Initialize the PPOTrainer
    trainer = PPOTrainer(
        model=model,
        args=training_args,
        train_dataset=data,
        data_collator=None,
        tokenizer=tokenizer,
    )
    
    # Call the train() function with the computed rewards
    train_result = trainer.train(rewards, resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save the trained model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/animal_guessing/animal_guessing.jsonl")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    main(args)