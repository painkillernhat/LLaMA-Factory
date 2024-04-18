import json
import os
from datasets import load_dataset, Dataset

os.environ["HF_DATASETS_CACHE"] = "/afs/cs.stanford.edu/u/sttruong/.cache"

def generate_records(input_file, output_file):
    with open(input_file) as file:
        data = json.load(file)

    all_records = []

    for item in data:
        instruction = item['instruction']
        input_value = item['input']
        output_value = item['output']
        system_value = item['system']
        history = item['history']

        records = []

        for i in range(len(history)):
            record = {
                'instruction': instruction,
                'input': history[i][0],
                'output': history[i][1],
                'system': system_value,
                'history': history[:i]
            }
            records.append(record)

        final_record = {
            'instruction': instruction,
            'input': input_value,
            'output': output_value,
            'system': system_value,
            'history': history
        }
        records.append(final_record)

        grouped_records = {
            'original_record': item,
            'prompted_record': records
        }

        all_records.append(grouped_records)

    prompt_records = []
    for group in all_records:
        prompt_records.extend(group['prompted_record'])

    # return prompted_records
    with open(output_file, 'w') as file:
        json.dump(prompt_records, file, indent=2)

def save_dataset_to_json(dataset, json_file_path):
    samples = [example for example in dataset]
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    with open(json_file_path, 'w') as json_file:
        json.dump(samples, json_file, indent=4)

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Iterative training and evaluation script")
    parser.add_argument("--sanity_check", type=str, default="False", help="Test")
    return parser.parse_args()

if __name__ == "__main__":

    current_directory = os.getcwd()
    
    input_file = os.path.join(current_directory, 'animal_guessing/animal_guessing.json')
    output_file = os.path.join(current_directory, 'animal_guessing_prompt.json')
    
    generate_records(input_file, output_file)
    dataset = load_dataset('json', data_files=output_file)

    # Load the original dataset
    args = parse_arguments()

    if args.sanity_check == 'True':
        train_test_dataset = dataset['train'].train_test_split(test_size=0.5)
        train_dataset = train_test_dataset['train']
        test_dataset = train_test_dataset['test']
    else:
        train_dataset = dataset['train']
        test_dataset = None

    splits = ['train', 'test']
    for split in splits:
        if split == 'train':
            dataset_split = train_dataset
        elif split == 'test' and test_dataset is not None:
            dataset_split = test_dataset
        else:
            continue

        output_dataset_path = os.path.join(current_directory, f"animal_guessing_{split}.json")
        save_dataset_to_json(dataset_split, output_dataset_path)
        print(f"{split}: {len(dataset_split)}")