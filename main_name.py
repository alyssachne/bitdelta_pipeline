import torch
from torch import nn
from torch import optim

import huggingface_hub
import transformers
from transformers import AutoModel, AutoTokenizer, DistilBertTokenizer, DistilBertForSequenceClassification, AutoModelForSequenceClassification
import time

import utils
from compressed_model import compress_diff
from dataset import get_dataset
import logging

logger = logging.getLogger('my_logger')


def check_model_layers(base_model, finetuned_model, trace=False):
    if trace:
        logger.info("Base Model:")
    # logger.info(base_model_config)
    base_dict = base_model.state_dict()
    if trace:
        for layer, weights in base_dict.items():
            logger.info(f"Layer: {layer}, Weights: {weights.size()}")

    if trace:
        logger.info("Finetuned Model:")
    # logger.info(base_model_config)
    finetuned_dict = finetuned_model.state_dict()
    if trace:
        for layer, weights in base_dict.items():
            logger.info(f"Layer: {layer}, Weights: {weights.size()}")

    # compare the weights
    for layer in base_dict.keys():
        if layer in finetuned_dict:
            if trace:
                logger.info(f"Layer: {layer}")
                logger.info(f"Base Model: {base_dict[layer].size()}")
                logger.info(f"Finetuned Model: {finetuned_dict[layer].size()}")
                logger.info("\n")
        else:
            logger.info(f"Layer: {layer} not found in finetuned model")
            return

    for layer in finetuned_dict.keys():
        if layer in base_dict:
            if trace:
                logger.info(f"Layer: {layer}")
                logger.info(f"Base Model: {base_dict[layer].size()}")
                logger.info(f"Finetuned Model: {finetuned_dict[layer].size()}")
                logger.info("\n")
        else:
            logger.info(f"Layer: {layer} not found in base model")
            return
        
    logger.info("Structure of both models are same")


def check_difference(base_model, finetuned_model, device):
    base_dict = base_model.state_dict()
    finetuned_dict = finetuned_model.state_dict()

    for layer in base_dict.keys():
        if "weight" not in str(layer) and "bias" not in str(layer):
            continue
        base_weights = base_dict[layer]
        finetuned_weights = finetuned_dict[layer]
        mean_diff = torch.mean(base_weights - finetuned_weights)
        logger.info(f"Layer {layer} has mean difference of {mean_diff}")


def weight_combine(base_model, finetuned_model, device):
    """
    Combine the weights of the models into two large 1D tensors.
    """
    base_dict = base_model.state_dict()
    finetuned_dict = finetuned_model.state_dict()

    # combine base weights
    base_weights = []
    for layer in base_dict.keys():

        if "weight" not in str(layer) and "bias" not in str(layer):
            continue

        curr_weights = base_dict[layer]
        curr_weights = curr_weights.view(-1).to(device)
        base_weights.append(curr_weights)
    base_weights = torch.cat(base_weights, dim=0).to(device)


    # combine finetuned weights
    finetuned_weights = []
    for layer in finetuned_dict.keys():
        if "weight" not in str(layer) and "bias" not in str(layer):
            continue

        curr_weights = finetuned_dict[layer]
        if curr_weights.is_meta:
            curr_weights = torch.zeros(curr_weights.size())
        curr_weights = curr_weights.view(-1).to(device)
        finetuned_weights.append(curr_weights)
    finetuned_weights = torch.cat(finetuned_weights, dim=0).to(device)


    return base_weights, finetuned_weights


def create_new_finetuned_weights(base_model, finetuned_model, device):
    base_weights, finetuned_weights = weight_combine(base_model, finetuned_model, device)
    logger.info("Amount of trainable weights: ", finetuned_weights.size())
    logger.info("\n")

    start_time = time.time()

    new_finetuned_weights = torch.where(finetuned_weights >= base_weights, 1.0, -1.0)
    positive_percentage = torch.sum(new_finetuned_weights.eq(1)).item() / new_finetuned_weights.size(0) * 100
    logger.info("Percentage of positive weights: ", positive_percentage)

    logger.info("Time taken: ", time.time() - start_time)

    return new_finetuned_weights


def create_new_finetuned_model(base_model, finetuned_model, finetuned_model_name, device):
    compressed_model = AutoModelForSequenceClassification.from_pretrained(finetuned_model_name).to(device)
    compress_diff(base_model, finetuned_model, compressed_model, device)
    return compressed_model


def compress(base_model_name, finetuned_model_name):

    device = utils.get_device()

    base_model = utils.load_model(base_model_name, device)

    finetuned_model = utils.load_model(finetuned_model_name, device)

    check_model_layers(base_model, finetuned_model)

    compressed_model = create_new_finetuned_model(base_model, finetuned_model, finetuned_model_name, device)

    return compressed_model


if __name__ == "__main__":
    compress()

