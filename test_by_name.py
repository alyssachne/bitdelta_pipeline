from datasets import load_metric
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

from dataset import get_dataset
from main_name import compress
import utils, compressed_model
import logging
import os
import argparse
from setup import *

def main(args):
    base_model_name = args.base_model
    finetuned_model_name = args.finetuned_model
    # dataset_name = args.dataset
    # it only supports glue for now
    dataset_name = "glue"
    if args.subdata:
        subdata = args.subdata
    else:
        subdata = None
    root_dir = f"saved/{finetuned_model_name}"

    logger = setup_logger(finetuned_model_name)

    logger.warning("For now the dataset can only comes from the hub.")
    dataset = setup_dataset(dataset_name, subdata, logger)
    if subdata in task_to_keys:
        sentence1_key, sentence2_key = setup_subdata_key(subdata)
    else:
        raise ValueError(f"Subdata {subdata} not supported.")


    tokenizer = setup_tokenizer(base_model_name, finetuned_model_name, logger)

    def encode(text):
        # tokenized_text = finetuned_tokenizer(text['sentence'], padding='max_length', truncation=True)
        # return tokenized_text
        kargs = (
            (text[sentence1_key],) if sentence2_key is None else (text[sentence1_key], text[sentence2_key])
        )
        result = tokenizer(*kargs, padding='max_length', truncation=True)
       
        return result

    ft_base_path = os.path.join(root_dir, 'ft_base.safetensors')
    ft_base = setup_and_save_original_model(finetuned_model_name, ft_base_path, logger)

    ft_compressed = compress(base_model_name, finetuned_model_name)
    logger.info("Model compressed.")

    accuracy_metric = load_metric('accuracy')
    metric = setup_metric(dataset_name, subdata)

    encoded_dataset = dataset.map(encode, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        other_metric = metric.compute(predictions=predictions, references=labels)
        return {"accuracy": accuracy, "other_metric": other_metric}

    # Evaluation arguments
    eval_args = TrainingArguments(
        output_dir='./results',
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        report_to="none", 
    )

    # Setup trainer
    trainer = Trainer(
        model=ft_compressed,
        args=eval_args,
        eval_dataset=encoded_dataset['validation'],
        compute_metrics=compute_metrics,
    )

    logger.info("Evaluation started.")

    results = trainer.evaluate()
    logger.info(results)

    logger.info("Evaluation finished.")

    compressed_path = os.path.join(root_dir, 'ft_compressed.safetensors')
    compressed_model.save_diff(ft_compressed, compressed_path)

    logger.info("Model saved.")
    test_model = compressed_model.load_diff(ft_base, compressed_path)

    logger.info("Saved model and loaded.")

    # Setup test trainer
    trainer = Trainer(
        model=test_model,
        args=eval_args,
        eval_dataset=encoded_dataset['validation'],
        compute_metrics=compute_metrics,
    )

    logger.info("Test evaluation started.")

    results = trainer.evaluate()

    logger.info(results)

    logger.info("Test evaluation finished.")

    ft_base_model_path = os.path.join(ft_base_path, "pytorch_model.bin")
    compress_rate = utils.compress_rate(ft_base_model_path, compressed_path)
    logger.info(f'The compression rate is {compress_rate}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the compressed model.")
    parser.add_argument("--base_model", type=str, help="The base model.")
    parser.add_argument("--finetuned_model", type=str, help="The finetuned model.") 
    parser.add_argument("--dataset", type=str, help="The dataset to evaluate performance, for now we only support glue.")
    parser.add_argument("--subdata", type=str, help="The sub-dataset to evaluate on if given.", required=False)
    args = parser.parse_args()

    main(args)
    

    

