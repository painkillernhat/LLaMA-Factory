import pandas as pd
import json
import random
import os

os.environ["HF_DATASETS_CACHE"] = "/afs/cs.stanford.edu/u/sttruong/.cache"

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


if __name__ == "__main__":
    all_data = []

    for index, row in data.iterrows():
        for col, question in questions.items():
            answer = generate_response(question, row[col])
            item = {
                "instruction": question,
                "input": "",
                "output": answer,
                "history": []
            }
            all_data.append(item)
        

    # Write the data to a JSON file
    with open('data/animal_rm.json', 'w') as outfile:
        json.dump(all_data, outfile, indent=4)
        