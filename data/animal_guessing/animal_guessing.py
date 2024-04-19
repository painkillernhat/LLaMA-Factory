import pandas as pd
import json
import random
import os

os.environ["HF_DATASETS_CACHE"] = "/afs/cs.stanford.edu/u/sttruong/.cache"

data = pd.read_csv("data/animal_guessing/animal.csv", sep=';')

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
    else:
        return "Yes." if value == 1 else "No."


if __name__ == "__main__":
    dataset = []

    for index, row in data.iterrows():
        animal_data = []
        history = []
        for col, question in questions.items():
            answer = generate_response(question, row[col])
            item = {
                "instruction": "Determine whether the user adopts or does not adopt the animal.",
                "input": question,
                "output": answer,
                "system": "",
                "history": history.copy()
            }
            history.append([question, answer])
            animal_data.append(item)
    
        # randomly decide to adopt or not adopt
        adopt_decision = random.choice(["I will adopt the animal.", "I will not adopt the animal."])
        if adopt_decision == "I will not adopt the animal.":
            outcome = "0"
        else:
            if row['domestic'] == 1:
                outcome = "1"  # home pet
            else:
                outcome = "-1"  # wild animal

        final_decision = {
            "instruction": "Determine whether the user adopts or does not adopt the animal.",
            "input": adopt_decision,
            "output": outcome,
            "system": "",
            "history": history
        }
        animal_data.append(final_decision)
        dataset.append(animal_data)

    with open('data/animal_guessing.json', 'w') as outfile:
        json.dump(dataset, outfile, indent=4)