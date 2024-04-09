import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

import huggingface_hub
import transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import os

from typing import Tuple


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
        ).to(device)
    
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


def get_device() -> torch.device:
    # check for cuda availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        device = torch.device("cpu")
        print("Using CPU")


def select_model(c: int):
    """
    Put your model names here.
    """
    if c == 1:
        base = "google-bert/bert-base-cased"
        finetuned = "piggyss/bert-finetuned-ner"
        return base, finetuned
    elif c == 2:
        return "CausalLM/14B"
        


def main(base_model_name, fintuned_model_name, device):
    base_model = load_model(base_model_name, device)
    base_tokenizer = load_tokenizer(base_model_name)
    print(base_model.config)
    print(base_tokenizer.vocab_size)

    fintuned_model = load_model(fintuned_model_name, device)
    fintuned_tokenizer = load_tokenizer(fintuned_model_name)
    print(fintuned_model.config)
    print(fintuned_tokenizer.vocab_size)


if __name__ == "__main__":
    # test

    # login to huggingface
    token = os.getenv("HF_TOKEN")
    huggingface_hub.login(token=token)

    base_model, fintuned_model = select_model(1)
    device = get_device()

    main(base_model, fintuned_model, device)
