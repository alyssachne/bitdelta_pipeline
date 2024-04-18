import os
import logging
from transformers import AutoModelForSequenceClassification
import utils
from dataset import get_dataset
from datasets import load_metric

def setup_logger(root_dir):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    logger_path = os.path.join(root_dir, 'output.txt')
    if os.path.exists(logger_path):
        os.remove(logger_path)
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add handler to the logger
    logger.addHandler(fh)

def setup_dataset(dataset_name, subdata, logger):
    if subdata:
        dataset = get_dataset(dataset_name, subdata)
    else:
        dataset = get_dataset(dataset_name)

    logger.info("Dataset loaded.")
    return dataset

def setup_tokenizer(model_name, logger):
    tokenizer = utils.load_tokenizer(model_name)
    logger.info("Tokenizer loaded.")
    return tokenizer

def setup_and_save_original_model(finetuned_model_name, ft_base_path, logger):
    ft_base = AutoModelForSequenceClassification.from_pretrained(finetuned_model_name)
    ft_base.save_pretrained(ft_base_path)
    logger.info("Original finetuned model saved.")
    return ft_base

def setup_metric(dataset, subset):
    if subset:
        metric = load_metric(dataset, subset)
    else:
        metric = load_metric(dataset)

    return metric

def setup_logger(finetuned_model_name):

    root_dir = f"saved/{finetuned_model_name}"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    #setup logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    logger_path = os.path.join(root_dir, 'output.txt')
    if os.path.exists(logger_path):
        os.remove(logger_path)
    fh = logging.FileHandler(logger_path)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add handler to the logger
    logger.addHandler(fh)

    return logger





