import json
import os
import datasets
from typing import Any, Dict, Generator, List, Tuple


_DESCRIPTION = "An example of dataset."
file_animal_path = "animal_guessing.jsonl"


class AnimalGuessingDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "instruction": datasets.Value("string"),
            "input": datasets.Value("string"),
            "output": datasets.Value("string"),
            "history": datasets.Sequence(datasets.Sequence(datasets.Value("string")))
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file_animal_path
                }
            )
        ]

    def _generate_examples(self, filepath: str) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        with open(filepath, "r", encoding="utf-8") as file:
            example_dataset = json.load(file)
        for key, example in enumerate(example_dataset):
            yield key, example


def save_dataset_to_json(dataset, json_file_path):
    samples = [example for example in dataset]
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    with open(json_file_path, 'w') as json_file:
        json.dump(samples, json_file, indent=4)


if __name__ == "__main__":
    dataset = datasets.load_dataset(__file__)

    dataset = dataset["train"].train_test_split(test_size=0.2)

    output_dir = os.path.join(os.path.dirname(__file__), "splits")
    os.makedirs(output_dir, exist_ok=True)

    splits = ['train', 'test']
    for split in splits:
        output_dataset_path = os.path.join(output_dir, f"animal_guessing_{split}.json")
        save_dataset_to_json(dataset[split], output_dataset_path)
        print(f"{split}: {len(dataset[split])}")