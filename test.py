import torch
from datasets import load_metric
from transformers import Trainer, TrainingArguments, DistilBertTokenizer, DistilBertForSequenceClassification

from dataset import get_dataset
from main import compress
import utils, compressed_model


choice = 3
dataset = get_dataset('glue', 'sst2')
print("Dataset loaded.")
base_model_name, finetuned_model_name = utils.select_model(choice)
finetuned_tokenizer = utils.load_tokenizer(finetuned_model_name)
print("Tokenizer loaded.")
base_model = DistilBertForSequenceClassification.from_pretrained(base_model_name)
finetuned_model = DistilBertForSequenceClassification.from_pretrained(finetuned_model_name)
base_model.save_pretrained("saved/ft_base")
print("Original finetuned model saved.")
ft_compressed = compress(choice)
print("Model compressed.")
print("\n")

#check compressed model architecture
print(base_model)
print("\n")
print(finetuned_model)


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

# Setup trainer
trainer = Trainer(
    model=ft_compressed,
    args=eval_args,
    eval_dataset=encoded_dataset['validation'],
    compute_metrics=compute_metrics,
)

print("Evaluation started.")

results = trainer.evaluate()
print(results)

print("Evaluation finished.")

compressed_model.save_diff(ft_compressed, "saved/ft_compressed.safetensors")

print("Model saved.")
print("\n")

test_model = compressed_model.load_diff(base_model, "saved/ft_compressed.safetensors")

print("Saved model and loaded.")

# Setup test trainer
trainer = Trainer(
    model=test_model,
    args=eval_args,
    eval_dataset=encoded_dataset['validation'],
    compute_metrics=compute_metrics,
)

print("Test evaluation started.")

results = trainer.evaluate()

print(results)

print("Test evaluation finished.")