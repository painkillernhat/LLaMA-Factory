import pandas as pd
import json
import random
import os
from sklearn.model_selection import train_test_split

# os.environ["HF_DATASETS_CACHE"] = "/afs/cs.stanford.edu/u/sttruong/.cache"

data = pd.read_csv("data/animal/animal.csv", sep=';')

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

def generate_response(question, value):
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

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Iterative training and evaluation script")
    parser.add_argument("--sanity_check", type=str, default="False", help="Test")
    return parser.parse_args()

def flatten_data(data):
    flat_list = []
    for animal_group in data:
        flat_list.extend(animal_group)
    return flat_list

def split_data(data, train_size=0.8):
    train_data, test_data = train_test_split(data, train_size=train_size, random_state=42)
    
    # flatten the data
    train_data_flat = flatten_data(train_data)
    test_data_flat = flatten_data(test_data)
    
    with open('data/animal_train.json', 'w') as f:
        json.dump(train_data_flat, f, indent=4)
    with open('data/animal_test.json', 'w') as f:
        json.dump(test_data_flat, f, indent=4)

if __name__ == "__main__":
    dataset = []

    for index, row in data.iterrows():
        animal_data = []
        history = []
        for col, question in questions.items():
            answer = generate_response(question, row[col])
            item = {
                "instruction": question,
                "input": "",
                "output": answer,
                "system": "",
                "history": history.copy()
            }
            history.append([question, answer])
            animal_data.append(item)
    
        # randomly decide to adopt or not adopt
        adopt_decision = random.choice(["I will adopt the animal.", "I will not adopt the animal."])
        if adopt_decision == "I will not adopt the animal.":
            if row['domestic'] == 1:
                outcome = "0"  # home pet
            else:
                outcome = "1"
            input_prompt = "If the animal is a home pet, the output is 0. If not, the output is 1."
        else:
            if row['domestic'] == 1:
                outcome = "1"  # home pet
            else:
                outcome = "-1"  # wild animal
            input_prompt = "If the animal is a home pet, the output is 1. If not, the output is -1."

        final_decision = {
            "instruction": adopt_decision,
            "input": input_prompt,
            "output": outcome,
            "system": "",
            "history": history
        }
        animal_data.append(final_decision)
        dataset.append(animal_data)

    args = parse_arguments()
 
    if args.sanity_check == 'True':
        split_data(dataset)
    else:
        with open('data/animal/animal_full.json', 'w') as outfile:
            json.dump(dataset, outfile, indent=4)