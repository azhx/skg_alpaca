#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import pickle
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer
from tqdm import tqdm
from bisect import bisect_right

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_instruction": (
        "### Input:\n{input}\n\n### Response:"
    )
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    pkl_path: str = field(default="data_v4_input_ids_labels.pkl", 
                          metadata={"help": "Path to the pickled version of tokenized data. Loading from this is faster. Must be in same directory as data_path"})
    has_instruction: bool = field(default=True, metadata={"help": "Whether we should use instructions for this dataset."})
    dataset_type: str = field(default="llama", metadata={"help": "Type of dataset. [llama, skg]"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, has_instruction: bool, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        # sources is all the input, up to "### Response: " targets is raw output
        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input, prompt_no_instr = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"], PROMPT_DICT["prompt_no_instruction"]
        sources = []
        targets = []
        for example in tqdm(list_data_dict):
            if has_instruction:
                sources.append(prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example))
            else:
                sources.append(prompt_no_instr.format_map(example))
            targets.append(f"{example['output']}{tokenizer.eos_token}")

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

class SupervisedSKGDataset(Dataset):

    def preprocess_and_tokenize(self, example, format_str, tokenizer):
        if example["text_in"]:
            pre_truncation = format_str.format(struct_in=example['struct_in'], text_in=example["text_in"]) + example["output"]
        else:
            pre_truncation = format_str.format(struct_in=example['struct_in']) + example["output"]

        # get the char index of when the struct starts and ends
        struct_start = pre_truncation.find(example['struct_in']) 
        struct_end = struct_start + len(example['struct_in'])
        output_start = pre_truncation.find(example['output'])

        if "xgen" in str(type(tokenizer)):            # if we are using xgen, we need to manually calculate the offsets
            seq_in = tokenizer(pre_truncation, return_tensors="pt")
            _, offsets = tokenizer.decode_with_offsets(seq_in['input_ids'][0])
            output_start_token = bisect_right(offsets, output_start)
            post_struct_token = bisect_right(offsets, struct_end)
            start_struct_token = bisect_right(offsets, struct_start)
        else:    
            seq_in = tokenizer(pre_truncation, return_tensors="pt")
            output_start_token = seq_in.char_to_token(output_start)
            post_struct_token = seq_in.char_to_token(struct_end)
            start_struct_token = seq_in.char_to_token(struct_start)
            assert seq_in.input_ids[:, -1] == 2 # eos token

        diff = max(seq_in.input_ids.shape[1] - tokenizer.model_max_length, 0)
        
        if (post_struct_token - start_struct_token <= diff and diff > 0):
            # if there is a struct and this is the case, we would have completely truncated the struct! 
            # If this is the case, we should just completely throw away the example
            # or, there is no struct, but the output is too long, so we should throw away the example
            return None
        
        struct_end_token = post_struct_token - diff
        input_ids = torch.concat((seq_in.input_ids[0,:struct_end_token], seq_in.input_ids[0,post_struct_token:]), dim=0)
        labels = input_ids.clone() # will modify this soon
        # truncating the struct by an offset of diff means the output will be shifted diff tokens earlier
        labels[:output_start_token - diff] = IGNORE_INDEX

        
        assert len(input_ids) == len(labels) <= tokenizer.model_max_length

        return dict(input_ids = input_ids, labels = labels)

    def __init__(self, data_path: str, has_instruction: bool, tokenizer: transformers.PreTrainedTokenizer, pkl_path: str):
        super(SupervisedSKGDataset, self).__init__()
        logging.warning("Loading data...")

        data_path_dir = os.path.dirname(data_path)
        full_pkl_path = os.path.join(data_path_dir, pkl_path)
        # if there is already a tokenized data file, then just load that
        if os.path.exists(full_pkl_path):
            logging.warning("Found tokenized data file, loading...")
            with open(full_pkl_path, "rb") as f:
                save = pickle.load(f)
            self.input_ids = save["input_ids"]
            self.labels = save["labels"]
            return


        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting and tokenizing inputs...")
        prompt_input, prompt_no_input, prompt_no_instr = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"], PROMPT_DICT["prompt_no_instruction"]
        self.input_ids = []
        self.labels = []
        self.final_data_identifiers = []
        for i, example in tqdm(enumerate(list_data_dict)):
            if has_instruction:
                format_str = prompt_input.format(instruction=example['instruction'], input=example['input_format'])

                data_dict = self.preprocess_and_tokenize(example, format_str, tokenizer)
            else:
                # In SKG, we always have inputs, but sometimes no instructions. In Alpaca, 
                # we may have instructions with no input but just response
                # if we are using prompt_no_instr, we are tuning exclusively on SKG data, which always has inputs
                format_str = prompt_no_instr.format(input=example['input_format'])
                data_dict = self.preprocess_and_tokenize(example, format_str, tokenizer)
            if data_dict is None:
                # warning
                logging.warning(f"Example {i} would have had its whole struct truncated, skipping...")
                continue
            self.input_ids.append(data_dict['input_ids'])
            self.labels.append(data_dict['labels'])
            self.final_data_identifiers.append(example['id'])
        # dump input_ids and labels into a huge pickle file
        save = {"input_ids": self.input_ids, "labels": self.labels}
        with open(full_pkl_path, "wb") as f:
            pickle.dump(save, f)

        # print the final dataset size
        logging.warning(f"Final dataset size: {len(self.input_ids)}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        #print("sizes", self.input_ids[i].shape, self.labels[i].shape)
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.dataset_type == "llama":
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, has_instruction=data_args.has_instruction)
    else:
        # dataset_type = "skg"
        train_dataset = SupervisedSKGDataset(tokenizer=tokenizer, data_path=data_args.data_path, has_instruction=data_args.has_instruction, pkl_path=data_args.pkl_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    print("loading tokenizer")
    special_tokens_dict = dict()
    if "xgen" in model_args.model_name_or_path:
        print("loading xgen")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            add_eos_token=True,
            trust_remote_code=True,
            pad_token="<|endoftext|>",
        )
        tokenizer.pad_token = tokenizer.eos_token        
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            add_eos_token=True,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk`_token"] = DEFAULT_UNK_TOKEN

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    print("done loading dataset")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

    print("done loading model")

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
