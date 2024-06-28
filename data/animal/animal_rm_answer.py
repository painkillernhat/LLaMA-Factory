import json

def convert(dataset):
    new_samples = []

    for conversation in dataset:
        output_list = []
        history = [] # keep prev questions

        for qa in conversation[:-1]:
            question = qa['instruction']
            correct_answer = qa['output']
            history.append({"question": question, "answer": correct_answer})

            if not output_list:  # if there are no previous outputs, duplicate the current correct answer
                new_qa = {
                    "instruction": question,
                    "input": qa['input'],
                    "output": [correct_answer, correct_answer],
                    "history": [{"question": h['question'], "answer": h['answer']} for h in history[:-1]]
                }
                new_samples.append(new_qa)
            else:
                # if there are previous outputs, create entries with each as a rejected answer
                for output in output_list:
                    new_qa = {
                        "instruction": question,
                        "input": qa['input'],
                        "output": [correct_answer, output],
                        "history": [{"question": h['question'], "answer": h['answer']} for h in history[:-1]]
                    }
                    new_samples.append(new_qa)

            output_list.append(correct_answer)

    return new_samples

if __name__ == "__main__":
    with open("data/animal/animal_full.json", 'r') as json_file:
        dataset = json.load(json_file)
    
    new_samples = convert(dataset)

    # for item in new_samples:
    #     print(item)
    with open("data/animal/animal_rm_answer.json", 'w') as file:
        json.dump(new_samples, file, indent=4)