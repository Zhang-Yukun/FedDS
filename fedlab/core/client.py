import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch

from selection.base_selector import BaseSelector
from utils.logger import logger



class ClientTrainer(object):
    def __init__(
            self,
            model,
            tokenizer,
            data_selector: BaseSelector,
            root_data_path: str,
            output_dir_root: str,
            config
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.data_selector = data_selector
        self.root_data_path = root_data_path
        with open(self.root_data_path, "r") as f:
            self.train_data = json.load(f)  # fixed -> for indexing all samples
        self.n_pool = len(self.train_data)
        self.labeled_idx = torch.zeros(self.n_pool, dtype=torch.bool)
        self.rd_to_labeled_idx = {}
        self.val_data = None
        self.output_dir_root = output_dir_root
        self.output_model_path = os.path.join(self.output_dir_root, "model")
        self.rd_output_model_path = os.path.join(self.output_model_path, "rd_0")
        self.config = config
        self.train_args = config.train_args
        self.lora_args = config.lora_args
        self.training_args = TrainingArguments(**self.train_args)


    def train(self, rd):
        self.rd_output_model_path = os.path.join(self.output_model_path, f"rd_{rd}")
        self.data_selector.updated_train_data(rd, self.tokenizer)
        self.train_args.output_dir = self.rd_output_model_path
        trainer = Trainer(model=self.model,
                          tokenizer=self.tokenizer,
                          args=self.training_args,
                          **self.data_selector.rd_labeled_data_module)
        trainer.train()
        logger.info(f"*** Round {rd} ** Training Done!")
        self.model.save_pretrained(self.rd_output_model_path)
        self.tokenizer.save_pretrained(self.rd_output_model_path)
        logger.info(f"*** Round {rd} ** Trainer State & Trained Model Saved To --> {self.rd_output_model_path} ***")

