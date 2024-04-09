import torch
from torch import nn
from torch import optim

import huggingface_hub
import transformers
from transformers import AutoModel, AutoTokenizer

import utils


def main():
    base_model_name, fintuned_model_name = utils.select_model(1)
    device = utils.get_device()

    base_model = utils.load_model(base_model_name, device)
    base_tokenizer = utils.load_tokenizer(base_model_name)

    fintuned_model = utils.load_model(fintuned_model_name, device)
    fintuned_tokenizer = utils.load_tokenizer(fintuned_model_name)


