import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

import huggingface_hub
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
import dataset
import os
import logging
from typing import Tuple
from compressed_model import BinaryDiff

logger = logging.getLogger('my_logger')

def load_model(model_name, device, memory_map=None) -> AutoModel:
    """
    Load the model using model name to device.

    If multiple gpus are used, a memory map needs to be provided.
    """
    # check for devices
    if device != 'auto' or not isinstance(device, list):
        return AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
    
    # if multi-gpu
    else:
        raise NotImplementedError("Multi-GPU support not implemented yet.")
    

def load_tokenizer(model_name):
    """
    Load the tokenizer using model name.
    """
    return AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True
    )


def load_config(model_name):
    """
    Load the model configuration using model name.
    """
    return AutoConfig.from_pretrained(
        model_name
    )


def get_device() -> torch.device:
    # check for cuda availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU:", torch.cuda.get_device_name())
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


def select_model(c: int):
    """
    Put your model names here.
    """
    if c == 1:
        base = "google/fnet-base"
        finetuned = "gchhablani/fnet-base-finetuned-sst2"
    elif c == 2:
        base = "google-bert/bert-base-cased"
        finetuned = "gchhablani/bert-base-cased-finetuned-sst2"
    elif c == 3:
        base = "distilbert/distilbert-base-uncased"
        finetuned = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    elif c == 4:
        base = "google/fnet-base"
        finetuned = "gchhablani/fnet-base-finetuned-sst2"
    return base, finetuned

def get_model_size(path):
    # Get the size of the file in bytes
    file_size = os.path.getsize(path)
    return file_size

def compress_rate(original_path, compressed_path):
    original_size = get_model_size(original_path)
    compressed_size = get_model_size(compressed_path)
    logger.info(f"Original size: {original_size} bytes")
    logger.info(f"Compressed size: {compressed_size} bytes")
    return compressed_size / original_size

    
# def save_diff(finetuned_compressed_model, save_dir):
#     diff_dict = {}

#     for name, module in finetuned_compressed_model.named_modules():
#         if isinstance(module, BinaryDiff):
#             logger.info(module.mask)
#             diff_dict[name + ".mask"] = module.mask.cpu()
#             logger.info(module.coeff)
#             diff_dict[name + ".coeff"] = module.coeff.cpu()

#     torch.save(diff_dict, save_dir)
        

def main(base_model_name, fintuned_model_name, device):
    base_model = load_model(base_model_name, device)
    base_tokenizer = load_tokenizer(base_model_name)
    logger.info("Base Model:")
    logger.info(base_model.config)
    logger.info(base_tokenizer.vocab_size)
    # logger.info(base_config)

    logger.info("Fintuned Model: ")
    fintuned_model = load_model(fintuned_model_name, device)
    fintuned_tokenizer = load_tokenizer(fintuned_model_name)
    logger.info(fintuned_model.config)
    logger.info(fintuned_tokenizer.vocab_size)

    # corrs, stddevs = find_corr_stddev(base_model, fintuned_model)
    # logger.info(corrs, stddevs)


if __name__ == "__main__":
    # test

    # login to huggingface
    token = os.getenv("HF_TOKEN")
    huggingface_hub.login(token=token)

    base_model_name, fintuned_model_name = select_model(1)
    device = get_device()

    main(base_model_name, fintuned_model_name, device)
