import json

def convert(dataset):
    new_samples = []

    for conversation in dataset:
        history = []

        # extract domestic status
        domestic_status = 0  # default
        for qa in conversation:
            if qa['instruction'] == "Is the animal domesticated?":
                domestic_status = 1 if qa['output'] == "Yes" else 0
                break

        for index, qa in enumerate(conversation[1:], start=1):
            question = qa['instruction']
            output = qa['output']

            # Store the full history up to the current question
            if index > 1:
                history.append({"question": conversation[index - 1]['instruction'], "answer": conversation[index - 1]['output']})

            decisions = [
                ("I will adopt the animal.", "1" if domestic_status == 1 else "-1"),
                ("I will not adopt the animal.", "1" if domestic_status == 1 else "0")
            ]
            for decision, actual_outcome in decisions:
                rejected_outcomes = ["1", "0", "-1"]
                rejected_outcomes.remove(actual_outcome)
                
                for rejected in rejected_outcomes:
                    new_samples.append({
                    "instruction": decision,
                    "input": "",
                    "output": [actual_outcome, rejected],
                    "history": list(history)
                })

    return new_samples

if __name__ == "__main__":
    with open("data/animal/animal_full.json", 'r') as json_file:
        dataset = json.load(json_file)
    
    new_samples = convert(dataset)
    # for item in new_samples:
    #     print(item)
    with open('data/animal/animal_rm_outcome.json', 'w') as file:
        json.dump(new_samples, file, indent=4)