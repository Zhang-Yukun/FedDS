import argparse
import os
import gc
import omegaconf
import torch
from peft import LoraConfig, get_peft_model

from selection import get_selector
from fedlab.core.local import LocTrainer
from utils.model import get_model_and_tokenizer, get_model
from utils.logger import logger



def main(args):
    rd = args.rd
    basic_config = omegaconf.OmegaConf.load(os.path.join(args.config_path, "basic_config.yaml"))
    p_configs = [
        omegaconf.OmegaConf.load(os.path.join(args.config_path, config_file)) for config_file in args.config_files
    ]
    config = omegaconf.OmegaConf.merge(basic_config, *p_configs)
    config["full_data_path"] = os.path.join(config["root_path"], "data", config["data_path"], config["data_name"])
    config["full_model_path"] = os.path.join(config["root_path"], "model", config["model_name"])
    config["full_output_path"] = os.path.join(
        config["root_path"], config["output_path"], "evol_res", config["model_name"], config['result_dir_name']
    )
    config["rd_data_path_root"] = os.path.join(
        config["full_output_path"], "data"
    )
    config["output_dir_root"] = os.path.join(
        config["full_output_path"], "output"
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
    if rd == 0:
        model, tokenizer = get_model_and_tokenizer(
            config["full_model_path"],
            model_max_length=config["model_max_length"]
        )
        if "lora_args" in config:
            peft_config = LoraConfig(**config["lora_args"])
            model = get_peft_model(model, peft_config)
            logger.info(f'*** Round {rd} ** New Model initialized with PEFT!')
        logger.info(f'*** Round {rd} ** New Model and Tokenizer initialized!')
        selector = get_selector(config["selector_name"])(
            root_data_path=config["full_data_path"],
            output_dir_root=config["full_output_path"],
            config=config
        )
        logger.info(f'*** Round {rd} ** New Selector initialized!')
        selector.init_labeled_data(num=config['init_label_num'])
        selector.save_labeled_idx()
        selector.save_rd_labeled_unlabeled_data(rd=rd)
        logger.info(f'*** Round {rd} ** labeled & unlabeled data saved!')
        trainer = LocTrainer(
            model=model,
            tokenizer=tokenizer,
            data_selector=selector,
            root_data_path=config["full_data_path"],
            output_dir_root=config["output_dir_root"],
            config=config
        )
        logger.info(f'*** Round {rd} ** New Trainer initialized!')
    else:
        model, tokenizer = get_model_and_tokenizer(
            config["full_model_path"],
            model_max_length=config["model_max_length"]
        )
        if "lora_args" in config:
            peft_config = LoraConfig(**config["lora_args"])
            model = get_peft_model(model, peft_config)
            logger.info(f'*** Round {rd} ** New Model initialized with PEFT!')
        logger.info(f'*** Round {rd} ** New Model and Tokenizer initialized!')
        selector = get_selector(config["selector_name"])(
            root_data_path=config["full_data_path"],
            output_dir_root=config["full_output_path"],
            config=config
        )
        logger.info(f'*** Round {rd} ** New Selector initialized!')
        selector.load_labeled_idx()
        selector.query(
            rd=rd,
            n=config["n_query"],
            tokenizer=tokenizer,
            use_model_path=os.path.join(config["output_dir_root"], "model", f"rd_{rd-1}")
        )
        selector.save_labeled_idx()
        torch.cuda.empty_cache()
        trainer = LocTrainer(
            model=model,
            tokenizer=tokenizer,
            data_selector=selector,
            root_data_path=config["full_data_path"],
            output_dir_root=config["output_dir_root"],
            config=config
        )
        logger.info(f'*** Round {rd} ** New Trainer initialized!')
    trainer.train(rd)
    torch.cuda.empty_cache()
    logger.info(f"*** Round {rd} ** Training Done !!!")
    logger.info("DiverseEvol Done ^_^")




if __name__ == "__main__":
    logger.info("DiverseEvol Start ^_^")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--config_files", type=str, nargs="+",required=True)
    parser.add_argument("--rd", type=int, default=0)
    main_args = parser.parse_args()
    main(main_args)
