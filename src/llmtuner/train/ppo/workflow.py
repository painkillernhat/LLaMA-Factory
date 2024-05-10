# Inspired by: https://github.com/lvwerra/trl/blob/main/examples/research_projects/stack_llama/scripts/rl_training.py
import math
import gc
import os
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
import copy
import torch
from tqdm import tqdm
from functools import partial
import json
from datasets import Dataset

from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits

from transformers import DataCollatorWithPadding

from ...data import get_dataset
from ...extras.callbacks import FixValueHeadModelCallback, LogCallback
from ...extras.misc import fix_valuehead_checkpoint
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..utils import create_ref_model, create_reward_model

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

from ...data.preprocess import preprocess_unsupervised_dataset, print_unsupervised_dataset_example
from ...data.template import get_template_and_fix_tokenizer

from .policy import Policy, PolicyPPOTrainer

def collate_fn(data):
    zipped = zip(data)
    return list(zipped)

def run_ppo():
    return

class Actor:
    def __init__(self, policy_model_args, policy_finetuning_args, training_args, generating_args):
        # self.bo_args = bo_args
        self.policy_model_args = policy_model_args
        self.policy_finetuning_args = policy_finetuning_args
        self.policy = Policy(self.policy_model_args, self.policy_finetuning_args)
        self.training_args = copy.deepcopy(training_args)
        self.training_args.output_dir = os.path.join(
            self.training_args.output_dir, "policy")

        self.generating_args = generating_args

        # self.algo_lookahead_steps = bo_args.algo_lookahead_steps

    def load_policy(self):
        self.policy.load()

    def unload_policy(self):
        self.policy.unload()
        
    def query(self, X, reward_model):
        # Query the next sequence
        breakpoint()  

    def convert_policy(self, data, n, num_repeat, group_size):
        # Initialize dictionary with categories containing appropriate data structures
        data_dict = {
            "prompt": [],
            "response": [],
            "reward": [],
            "system": [],
            "tools": []
        }
        batch_size = n * num_repeat
    
        # Iterate through data in steps of batch_size
        for i in range(0, len(data), batch_size):
            for offset in range(batch_size):
                # Calculate the indices for the current group
                group_indices = [i + offset + j * batch_size for j in range(group_size)]
                # Check if the last index in the group is within data range
                if group_indices[-1] >= len(data):
                    continue  # Skip incomplete groups
    
                # Focus on the most detailed record (the last in the sequence)
                detailed_record = data[group_indices[-1]]
    
                # Construct the prompt from the most detailed record's history and instruction
                prompt = []
                for h in detailed_record['history']:
                    prompt.append({"role": "user", "content": h[0]})
                    prompt.append({"role": "assistant", "content": h[1] if h[1] else detailed_record['instruction']})
    
                # Construct the response from the output of the most detailed record
                response = [{"role": "user", "content": detailed_record['output']}]
    
                # Append the data to the main dataset dictionary
                data_dict["prompt"].extend([prompt])
                data_dict["response"].extend([response])
                data_dict["reward"].extend([[detailed_record['reward']]])
                data_dict["system"].extend([[""]])
                data_dict["tools"].extend([[""]])
    
        return data_dict

    def train_policy(
        self,
        data_path,
        data_args,
        callbacks: Optional[List["TrainerCallback"]] = None,
    ) -> None:
        template = get_template_and_fix_tokenizer(self.policy.tokenizer, data_args.template)
        preprocess_func = partial(
            preprocess_unsupervised_dataset, tokenizer=self.policy.tokenizer, template=template, data_args=data_args
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=self.policy.tokenizer)
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
        
        with open(data_path, 'r') as file:
            data = json.load(file)  
        dataset = self.convert_policy(data, 99, 17, 3 - 1)
        
        file_path = f"{data_args.dataset_dir}/{data_args.dataset}_demo.jsonl"
        json_data = json.dumps(dataset, indent=4)
        with open(file_path, 'w') as json_file:
            json_file.write(json_data)

        dataset = Dataset.from_dict(dataset)  # Convert dictionary to a Dataset object
        dataset = dataset.map(preprocess_func, batched=True, remove_columns=["prompt", "response", "system", "tools", "reward"], **kwargs)
        
        # print_function(dataset)

        # Set callbacks
        callbacks = [LogCallback()] if callbacks is None else callbacks

        # use left-padding in generation while using right-padding in training
        self.policy.tokenizer.padding_side = "left"
        self.training_args.remove_unused_columns = False  # important for pairwise dataset

        # Create reference model
        ref_model = create_ref_model(
            self.policy_model_args, self.policy_finetuning_args, add_valuehead=True)

        # data_collator = DataCollatorWithPadding(tokenizer=self.policy.tokenizer)

        # Initialize our Trainer
        ppo_trainer = PolicyPPOTrainer(
            model_args=self.policy_model_args,
            training_args=self.training_args,
            finetuning_args=self.policy_finetuning_args,
            generating_args=self.generating_args,
            callbacks=callbacks + [FixValueHeadModelCallback()],
            model=self.policy.model,
            ref_model=ref_model,
            tokenizer=self.policy.tokenizer,
            dataset=dataset,
            data_collator=collate_fn,
        )

        # Training
        if self.training_args.do_train:
            ppo_trainer.ppo_train(
                resume_from_checkpoint=self.training_args.resume_from_checkpoint)
            ppo_trainer.save_model()
            if self.training_args.should_save:
                fix_valuehead_checkpoint(
                    self.policy.model, self.training_args.output_dir, self.training_args.save_safetensors)
            ppo_trainer.save_state()  # must be called after save_model to have a folder
            if ppo_trainer.is_world_process_zero() and self.policy_finetuning_args.plot_loss:
                plot_loss(self.training_args.output_dir,
                          keys=["loss", "reward"])
