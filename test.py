import torch
from datasets import load_metric
from transformers import Trainer, TrainingArguments, DistilBertTokenizer, DistilBertForSequenceClassification, AutoModelForSequenceClassification

from dataset import get_dataset
from main import compress
import utils, compressed_model
import logging
import os

choice = 4

base_model_name, finetuned_model_name = utils.select_model(choice)
root_dir = f"saved/{finetuned_model_name}"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

#setup logger object
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

dataset = get_dataset('glue', "sst2")
logger.info("Dataset loaded.")

finetuned_tokenizer = utils.load_tokenizer(base_model_name)
logger.info("Tokenizer loaded.")
ft_base = AutoModelForSequenceClassification.from_pretrained(finetuned_model_name)
ft_base_path = os.path.join(root_dir, 'ft_base.safetensors')
ft_base.save_pretrained(ft_base_path)
logger.info("Original finetuned model saved.")
ft_compressed = compress(choice)
logger.info("Model compressed.")
logger.info("\n")

def encode(text):
    tokenized_text = finetuned_tokenizer(text['sentence'], padding='max_length', truncation=True)
    return tokenized_text

encoded_dataset = dataset.map(encode, batched=True)
# Metric and compute_metrics function
metric = load_metric('glue', "sst2")
# Load the accuracy metric
accuracy_metric = load_metric('accuracy')

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
logger.info("\n")

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

logger.info(f'The compression rate is {utils.compress_rate(os.path.join(ft_base_path, "pytorch_model.bin"), compressed_path)}')