from datasets import load_metric
from transformers import Trainer, TrainingArguments

from main_name import compress
import utils, compressed_model
import os
import argparse
from setup import *

def evaluate_model(
        base_model_name, 
        finetuned_model_name, 
        logger, 
        root_dir,
        subdata,
        dataset_name
    ):
    # it only supports glue for now
    dataset_name = "glue"
    if subdata:
        subdata = subdata
    else:
        subdata = None

    logger.warning("For now the dataset can only comes from the hub.")
    dataset = setup_dataset(dataset_name, subdata, logger)
    if subdata in task_to_keys:
        sentence1_key, sentence2_key = setup_subdata_key(subdata)
    else:
        raise ValueError(f"Subdata {subdata} not supported.")


    tokenizer = setup_tokenizer(base_model_name, finetuned_model_name, logger)

    def encode(text):
        kargs = (
            (text[sentence1_key],) if sentence2_key is None else (text[sentence1_key], text[sentence2_key])
        )
        result = tokenizer(*kargs, padding='max_length', truncation=True)
       
        return result

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

    ft_compressed = compress(base_model_name, finetuned_model_name)
    logger.info("Model compressed.")

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    compressed_path = os.path.join(root_dir, 'ft_compressed.safetensors')
    compressed_model.save_diff(ft_compressed, compressed_path)

    logger.info("Model saved.")
    base = setup_model(base_model_name)
    test_model = compressed_model.load_diff(base, compressed_path)

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

def main(args):
    base_model_name = args.base_model
    finetuned_model_path = args.finetuned_models
    
    ft_models = setup_ft_models(finetuned_model_path)

    for finetuned_model_name in ft_models.keys():
        root_dir = f"saved_multi/{finetuned_model_name}"

        logger = setup_logger(finetuned_model_name, root_dir)

        dataset_name, subdata = ft_models[finetuned_model_name]

        evaluate_model(
            base_model_name, 
            finetuned_model_name, 
            logger, 
            root_dir,
            subdata,
            dataset_name)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the compressed model.")
    parser.add_argument("--base_model", type=str, help="The base model.", default="google-bert/bert-base-uncased")
    parser.add_argument("--finetuned_models", type=str, help="The path to fine-tuned models' information.", default="/h/u6/c9/01/cheny845/csc413/bitdelta/ft_model.json")
    args = parser.parse_args()

    main(args)
    

    

