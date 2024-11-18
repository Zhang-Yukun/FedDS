from dataclasses import dataclass
from typing import Sequence
from copy import deepcopy
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from torch import Tensor
import torch

from .logger import logger
from .consts import PROMPT_DICT, LLAMA_IGNORE_INDEX
from .iostream import jload
from .args import SFTDataArguments


def _tokenize_fn(
        strings: Sequence[str],
        tokenizer: PreTrainedTokenizer,
        mode: str = "train"
) -> dict:
    """
    Tokenize the strings for Auto-regressive Supervised Fine-tuning (SFT) tasks.
    Args:
        strings (List[str]): List of strings to tokenize.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
        mode (str): Task mode. Default is 'train'. Train mode will truncate the input to the model's max length.

    Returns:
        dict: Dictionary containing the tokenized input_ids, labels, input_ids_lens, and labels_lens.

    Examples:
        >>> _tokenize_fn(["Hello, world!", "How are you?"], tokenizer)
        {'input_ids': [[101, 7592, 1010, 2088, 999, 102], [101, 2129, 2024, 2017, 1029, 102]],
         'labels': [[-100, 7592, 1010, 2088, 999, 102], [-100, 2129, 2024, 2017, 1029, 102]],
         'input_ids_lens': [6, 6], 'labels_lens':
    """
    truncation = True if mode == "train" else False
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,  # usable when mode == train
            truncation=truncation,
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
        tokenizer: PreTrainedTokenizer,
        mode: str = "train"
) -> dict:
    """
    Preprocess the data by tokenizing.

    Args:
        sources (List[str]): List of source strings.
        targets (List[str]): List of target strings (supervision signal).
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
        mode (str): Task mode. Default is 'train', which will concatenate the source and target strings as training data
            and mask out the source part in the labels.
    Returns:
        dict: Dictionary containing the tokenized input_ids and labels.

    """
    if mode == "train":
        examples = [s + t for s, t in zip(sources, targets)]
        logger.debug("Below is the first example in the examples list: >>>")
        logger.debug(examples[0])
        logger.debug("Above is the first example in the examples list: <<<")
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = LLAMA_IGNORE_INDEX
    else:
        sources_tokenized = _tokenize_fn(sources, tokenizer)
        targets_tokenized = _tokenize_fn(targets, tokenizer, mode)  # fix truncated label
        input_ids = sources_tokenized["input_ids"]
        labels = targets_tokenized["input_ids"]
    return dict(input_ids=input_ids, labels=labels)


## DATASETS / DATALOADER
class SupervisedDataset(Dataset):
    """Dataset for sft."""
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: SFTDataArguments
    ):
        super(SupervisedDataset, self).__init__()
        logger.info("Loading data...")
        list_data_dict = jload(data_args.data_path)
        logger.info(f"making supervised_dataset -> jload('{data_args.data_path}') SUCCESSFULLY")
        logger.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        logger.debug("Below is the first source in the sources list: >>>")
        logger.debug(sources[0])
        logger.debug("Above is the first source in the sources list: <<<")
        logger.debug("Below is the first target in the targets list: >>>")
        logger.debug(targets[0])
        logger.debug("Above is the first target in the targets list: <<<")
        logger.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for sft."""
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=LLAMA_IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: PreTrainedTokenizer, data_args) -> dict:
    """Make dataset and collator for SFT."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)