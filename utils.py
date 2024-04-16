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

from typing import Tuple
from compressed_model import BinaryDiff


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
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def select_model(c: int):
    """
    Put your model names here.
    """
    if c == 1:
        base = "google-bert/bert-base-cased"
        finetuned = "piggyss/bert-finetuned-ner"
    elif c == 2:
        return "CausalLM/14B"
    elif c == 3:
        base = "distilbert/distilbert-base-uncased"
        finetuned = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    elif c == 4:
        base = 'distilbert/distilroberta-base'
        finetuned = 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'
    return base, finetuned


# def save_diff(finetuned_compressed_model, save_dir):
#     diff_dict = {}

#     for name, module in finetuned_compressed_model.named_modules():
#         if isinstance(module, BinaryDiff):
#             print(module.mask)
#             diff_dict[name + ".mask"] = module.mask.cpu()
#             print(module.coeff)
#             diff_dict[name + ".coeff"] = module.coeff.cpu()

#     torch.save(diff_dict, save_dir)
        

def main(base_model_name, fintuned_model_name, device):
    base_model = load_model(base_model_name, device)
    base_tokenizer = load_tokenizer(base_model_name)
    print("Base Model:")
    print(base_model.config)
    print(base_tokenizer.vocab_size)
    # print(base_config)

    print("Fintuned Model: ")
    fintuned_model = load_model(fintuned_model_name, device)
    fintuned_tokenizer = load_tokenizer(fintuned_model_name)
    print(fintuned_model.config)
    print(fintuned_tokenizer.vocab_size)

    # corrs, stddevs = find_corr_stddev(base_model, fintuned_model)
    # print(corrs, stddevs)


if __name__ == "__main__":
    # test

    # login to huggingface
    token = os.getenv("HF_TOKEN")
    huggingface_hub.login(token=token)

    base_model_name, fintuned_model_name = select_model(1)
    device = get_device()

    main(base_model_name, fintuned_model_name, device)
