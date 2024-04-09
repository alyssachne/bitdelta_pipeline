import torch
from torch import nn
from torch import optim

import huggingface_hub
import transformers
from transformers import AutoModel, AutoTokenizer

import model


def check_model_layers(base_model, finetuned_model, trace=False):
    if trace:
        print("Base Model:")
    # print(base_model_config)
    base_dict = base_model.state_dict()
    if trace:
        for layer, weights in base_dict.items():
            print(f"Layer: {layer}, Weights: {weights.size()}")

    if trace:
        print("Finetuned Model:")
    # print(base_model_config)
    finetuned_dict = finetuned_model.state_dict()
    if trace:
        for layer, weights in base_dict.items():
            print(f"Layer: {layer}, Weights: {weights.size()}")

    # compare the weights
    for layer in base_dict.keys():
        if layer in finetuned_dict:
            if trace:
                print(f"Layer: {layer}")
                print(f"Base Model: {base_dict[layer].size()}")
                print(f"Fintuned Model: {finetuned_dict[layer].size()}")
                print("\n")
        else:
            print(f"Layer: {layer} not found in fintuned model")
            return

    for layer in finetuned_dict.keys():
        if layer in base_dict:
            if trace:
                print(f"Layer: {layer}")
                print(f"Base Model: {base_dict[layer].size()}")
                print(f"Fintuned Model: {finetuned_dict[layer].size()}")
                print("\n")
        else:
            print(f"Layer: {layer} not found in base model")
            return
        
    print("Structure of both models are same")


def check_difference(base_model, finetuned_model):
    base_dict = base_model.state_dict()
    finetuned_dict = finetuned_model.state_dict()

    for layer in base_dict.keys():
        base_weights = base_dict[layer]
        finetuned_weights = finetuned_dict[layer]
        mean_diff = torch.mean(base_weights - finetuned_weights)
        print(f"Layer {layer} has mean difference of {mean_diff}")


def main():
    base_model_name, fintuned_model_name = model.select_model(1)
    device = model.get_device()

    base_model = model.load_model(base_model_name, device)
    base_tokenizer = model.load_tokenizer(base_model_name)
    base_model_config = base_model.config

    fintuned_model = model.load_model(fintuned_model_name, device)
    fintuned_tokenizer = model.load_tokenizer(fintuned_model_name)
    fintuned_model_config = fintuned_model.config

    check_model_layers(base_model, fintuned_model)

    check_difference(base_model, fintuned_model)

if __name__ == "__main__":
    main()

