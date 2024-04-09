import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

import huggingface_hub
import transformers
from transformers import AutoModel, AutoTokenizer
import os


# load the model
def load_model():
    model = AutoModel.from_pretrained("CausalLM/14B")
    tokenizer = AutoTokenizer.from_pretrained("CausalLM/14B")
    print(model.config)
    return model, tokenizer


def main():
    model, tokenizer = load_model()


if __name__ == "__main__":
    # login to huggingface
    token = os.getenv("HF_TOKEN")
    huggingface_hub.login(token=token)
    main()
