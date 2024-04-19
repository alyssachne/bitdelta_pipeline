import os
import logging
from transformers import AutoModelForSequenceClassification, DistilBertForSequenceClassification
import utils
from dataset import get_dataset
from datasets import load_metric
import json

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# def setup_logger(root_dir):
#     logger = logging.getLogger('my_logger')
#     logger.setLevel(logging.DEBUG)

#     logger_path = os.path.join(root_dir, 'output.txt')
#     if os.path.exists(logger_path):
#         os.remove(logger_path)
#     fh = logging.FileHandler(logger_path)
#     fh.setLevel(logging.DEBUG)

#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     fh.setFormatter(formatter)

#     # add handler to the logger
#     logger.addHandler(fh)

def setup_dataset(dataset_name, subdata, logger):
    if subdata:
        dataset = get_dataset(dataset_name, subdata)
        logger.info(f"Dataset {dataset_name}-{subdata} loaded.")
    else:
        dataset = get_dataset(dataset_name)
        logger.info(f"Dataset {dataset_name} loaded.")
    
    return dataset

def setup_tokenizer(model_name, finetuned_model_name, logger):
    try:
        tokenizer = utils.load_tokenizer(finetuned_model_name)
    except:
        tokenizer = utils.load_tokenizer(model_name)

    logger.info("Tokenizer loaded.")
    return tokenizer

def setup_and_save_original_model(finetuned_model_name, ft_base_path, logger):
    ft_base = setup_model(finetuned_model_name)
    ft_base.save_pretrained(ft_base_path)
    logger.info(f"Original finetuned model {finetuned_model_name} saved.")
    return ft_base

def setup_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model

def setup_metric(dataset, subset):
    if subset:
        metric = load_metric(dataset, subset)
    else:
        metric = load_metric(dataset)

    return metric

def setup_subdata_key(subdata):

    sentence1_key, sentence2_key = task_to_keys[subdata]

    return sentence1_key, sentence2_key

def setup_ft_models(ft_model_json):
    # load ft model info from json file
    ft_model_info = json.load(open(ft_model_json))
    ft_models = {}
    for key in ft_model_info:
        info = ft_model_info[key]
        ft_model = info["ft_model"]
        dataset_name = info["dataset"]
        subdata = info["subdata"]
        ft_models[ft_model] = (dataset_name, subdata)
    
    return ft_models    

def setup_logger(root_dir):

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





