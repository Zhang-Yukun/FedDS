import os
import json
import pickle
import torch

from copy import deepcopy
from peft import PeftModel

from utils.logger import logger
from utils.iostream import jdump
from utils.data import make_supervised_data_module
from utils.args import SFTDataArguments
from utils.model import get_model
from utils.functional import get_embeddings


class BaseSelector(object):
    def __init__(
            self,
            root_data_path: str,
            output_dir_root: str,
            config
    ):
        self.root_data_path = root_data_path
        self.output_dir_root = output_dir_root
        self.config = config
        self.output_data_path = os.path.join(self.output_dir_root, "data")
        self.rd_output_data_dir = os.path.join(self.output_data_path, f"rd_0")
        os.makedirs(self.output_data_path, exist_ok=True)
        with open(self.root_data_path, "r") as f:
            self.train_data = json.load(f)  # fixed -> for indexing all samples
        self.n_pool = len(self.train_data)
        self.labeled_idx = torch.zeros(self.n_pool, dtype=torch.bool)
        self.rd_to_labeled_idx = {}
        self.rd_labeled_data_module = dict()
        self.rd_unlabeled_data_module = dict()

    def init_labeled_data(self, num=-1):
        if num == -1:
            self.labeled_idx[:] = True
            self.rd_to_labeled_idx[0] = torch.arange(self.n_pool)
        else:
            temp_idxes = torch.randperm(self.n_pool)
            self.labeled_idx[temp_idxes[:num]] = True
            self.rd_to_labeled_idx[0] = temp_idxes[:num].sort().values

    def set_labeled_data(self, labeled_idx, rd_to_labeled_idx):
        self.labeled_idx = labeled_idx
        self.rd_to_labeled_idx = rd_to_labeled_idx

    def save_labeled_idx(self, client=None):
        client_dict = {
            "labeled_idx": deepcopy(self.labeled_idx),
            "rd_to_labeled_idx": deepcopy(self.rd_to_labeled_idx)
        }
        if client is None:
            with open(
                    os.path.join(self.config['output_dir_root'], "client_dict.pkl"), "wb"
            ) as f:
                pickle.dump(client_dict, f)
        else:
            with open(
                    os.path.join(self.config['output_dir_root'], f"client_{client}", "client_dict.pkl"), "wb"
            ) as f:
                pickle.dump(client_dict, f)

    def load_labeled_idx(self, client=None):
        if client is None:
            with open(
                    os.path.join(self.config['output_dir_root'], "client_dict.pkl"), "rb"
            ) as f:
                client_dict = pickle.load(f)
        else:
            with open(
                    os.path.join(self.config['output_dir_root'], f"client_{client}", "client_dict.pkl"), "rb"
            ) as f:
                client_dict = pickle.load(f)
        self.labeled_idx = client_dict["labeled_idx"]
        self.rd_to_labeled_idx = client_dict["rd_to_labeled_idx"]

    def save_rd_labeled_unlabeled_data(self, rd):
        labeled_idx = torch.arange(self.n_pool)[self.labeled_idx.bool()]  # self.labeled_idx -> kept updated
        unlabeled_idx = torch.arange(self.n_pool)[~self.labeled_idx.bool()]
        rd_labeled_idx = self.rd_to_labeled_idx[rd]  # self.rd_to_labeled_idx -> kept track
        assert labeled_idx.equal(rd_labeled_idx)  # check -> self.labeled_idx gets properly updated like tracked.
        # query self.train_data -> current labeled & unlabeled data
        labeled_data_json_format = [self.train_data[_] for _ in labeled_idx] # list of dict
        unlabeled_data_json_format = [self.train_data[_] for _ in unlabeled_idx]
        logger.info(f"*** Round {rd} ** labeled_idx: {labeled_idx}")
        rd_labeled_data_path = os.path.join(self.output_data_path, f"rd_{rd}_labeled.json")
        rd_unlabeled_data_path = os.path.join(self.output_data_path, f"rd_{rd}_unlabeled.json")
        logger.info(f"*** Round {rd} ** Saving labeled & unlabeled data to {self.output_data_path}")
        jdump(labeled_data_json_format, rd_labeled_data_path)
        jdump(unlabeled_data_json_format, rd_unlabeled_data_path)

    def updated_train_data(self, rd, tokenizer):
        self.rd_output_data_dir = os.path.join(self.output_data_path, f"rd_{rd}")
        sft_args = SFTDataArguments(os.path.join(self.output_data_path, f"rd_{rd}_labeled.json"))
        self.rd_labeled_data_module = make_supervised_data_module(
            tokenizer=tokenizer,
            data_args=sft_args
        )

    def update_unlabeled_data(self, rd, tokenizer):
        self.rd_output_data_dir = os.path.join(self.output_data_path, f"rd_{rd}")
        sft_args = SFTDataArguments(os.path.join(self.output_data_path, f"rd_{rd}_unlabeled.json"))
        self.rd_unlabeled_data_module = make_supervised_data_module(
            tokenizer=tokenizer,
            data_args=sft_args
        )

    def update_rd(self, rd, add_labeled_idx):
        self.labeled_idx[add_labeled_idx.to(self.labeled_idx.device)] = True
        labeled_idx = torch.arange(self.n_pool)[self.labeled_idx.bool()]
        self.rd_to_labeled_idx[rd] = labeled_idx  # keep track of each round's labeled data
        self.save_rd_labeled_unlabeled_data(rd=rd)  # save labeled & unlabeled data for each round

    def get_embeddings_all_data(self, rd, tokenizer, use_model_path):
        """compute last hidden states for full dataset -> distance-based schedules"""
        sft_args = SFTDataArguments(self.root_data_path)
        model = get_model(self.config["full_model_path"])
        if "lora_args" in self.config:
            model = PeftModel.from_pretrained(model, use_model_path)
            # model.merge_and_unload()
        logger.info(f"*** Round {rd} ** Computing embeddings with model from {use_model_path}!")
        all_data = make_supervised_data_module(
            tokenizer=tokenizer,
            data_args=sft_args
        )
        logger.info(f'*** Round {rd} ** Trained Model loaded!')
        return get_embeddings(data=all_data, model=model)

    def query(self, rd=0, n=0, tokenizer=None, use_model_path=None):
        pass
