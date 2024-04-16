import torch
from datasets import load_metric
from transformers import Trainer, TrainingArguments, DistilBertTokenizer, DistilBertForSequenceClassification

from dataset import get_dataset
from main import compress
import utils


choice = 3
dataset = get_dataset("glue", 'sst2')
print("Dataset loaded.")
base_model_name, finetuned_model_name = utils.select_model(choice)
finetuned_tokenizer = utils.load_tokenizer(finetuned_model_name)
print("Tokenizer loaded.")
compressed_model = compress(choice)
print("Model compressed.")
print("\n")

# for name, module in compressed_model.named_modules():
#     for submodule_name, submodule in module.named_children():
#         print(f"Module: {name}, Submodule: {submodule_name, submodule}")

def encode(text):
    tokenized_text = finetuned_tokenizer(text['sentence'], padding='max_length', truncation=True)
    return tokenized_text

encoded_dataset = dataset.map(encode, batched=True)

# Metric and compute_metrics function
metric = load_metric('glue', 'sst2')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Evaluation arguments
eval_args = TrainingArguments(
    output_dir='./results',
    do_train=False,
    do_eval=True,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
)

# Setup Trainer
trainer = Trainer(
    model=compressed_model,
    args=eval_args,
    eval_dataset=encoded_dataset['validation'],
    compute_metrics=compute_metrics,
)

print("Evaluation started.")

results = trainer.evaluate()
print(results)

utils.save_diff(compressed_model, "diff.pt")

print("Evaluation finished.")