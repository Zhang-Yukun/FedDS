import argparse
import os
import omegaconf
import torch
from peft import LoraConfig, get_peft_model

from utils.model import get_model_and_tokenizer
from utils.logger import logger
from utils.iostream import load_adapters_to_tensor
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool



def main(args):
    rd = args.rd
    basic_config = omegaconf.OmegaConf.load(os.path.join(args.config_path, "basic_config.yaml"))
    p_configs = [
        omegaconf.OmegaConf.load(os.path.join(args.config_path, config_file)) for config_file in args.config_files
    ]
    config = omegaconf.OmegaConf.merge(basic_config, *p_configs)
    config["full_data_path"] = os.path.join(config["root_path"], "data", config["data_path"])
    config["full_model_path"] = os.path.join(config["root_path"], "model", config["model_name"])
    config["full_output_path"] = os.path.join(
        config["root_path"], config["output_path"], "evol_res", config["model_name"], config['result_dir_name']
    )
    config["rd_data_path_root"] = os.path.join(
        config["full_output_path"], "data", "server"
    )
    config["output_dir_root"] = os.path.join(
        config["full_output_path"], "output", "server"
    )
    os.makedirs(config["rd_data_path_root"], exist_ok=True)
    os.makedirs(config["output_dir_root"], exist_ok=True)
    logger.add(
        os.path.join(config["full_output_path"], "logger.log"),
        level="INFO",
        enqueue=True
    )
    logger.info('Configuration loaded!')
    logger.info(omegaconf.OmegaConf.to_yaml(config))
    omegaconf.OmegaConf.save(config, os.path.join(config["full_output_path"], "config.yaml"))
    logger.info(f"*** Round {rd} ======================================================================================================")
    model_list = []
    model, tokenizer = get_model_and_tokenizer(
        config["full_model_path"],
        model_max_length=config["model_max_length"]
    )
    if "lora_args" in config:
        peft_config = LoraConfig(**config["lora_args"])
        model = get_peft_model(model, peft_config)
        logger.info(f'*** Round {rd} ** New Model initialized with PEFT for Server!')
    logger.info(f'*** Round {rd} ** New Model and Tokenizer initialized for Server!')
    for client in range(config["n_clients"]):
        model_list.append(load_adapters_to_tensor(
            os.path.join(config["full_output_path"], "output", f"client_{client}", "model", f"rd_{rd}", "adapter_model.safetensors")
        ))
        logger.info(f'*** Round {rd} ** Model loaded for Client {client}!')
    serialized_parameters = Aggregators.fedavg_aggregate(model_list)
    SerializationTool.deserialize_model(model, serialized_parameters, use_peft=True)
    logger.info(f'*** Round {rd} ** Model Aggregated for Server!')
    model.save_pretrained(os.path.join(config["output_dir_root"], "model", f"rd_{rd}"))




if __name__ == "__main__":
    logger.info("DiverseEvol Start ^_^")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_files", type=str, nargs="+",required=True)
    parser.add_argument("--rd", type=int, required=True)
    main_args = parser.parse_args()
    main(main_args)
