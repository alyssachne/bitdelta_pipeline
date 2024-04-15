import torch
from datasets import load_metric
from transformers import Trainer, TrainingArguments

from dataset import get_dataset
from main import compress
import utils


dataset = get_dataset('glue', 'sst2')
print("Dataset loaded.")
base_model_name, finetuned_model_name = utils.select_model(1)
finetuned_tokenizer = utils.load_tokenizer(finetuned_model_name)
print("Tokenizer loaded.")
compressed_model = compress()
print("Model compressed.")
print("\n")

def encode(text):
    tokenized_text = finetuned_tokenizer(text['sentence'], padding='max_length', truncation=True)
    return tokenized_text

encoded_dataset = dataset.map(encode, batched=True)

metric = load_metric('glue', 'sst2')
def compute_metrics(eval_pred):
    logits = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_eval_batch_size=64,
    no_cuda=not torch.cuda.is_available(),
    do_train=False,  # we're only doing evaluation
    do_eval=True,
)

print("\n")

trainer = Trainer(
    model=compressed_model,
    args=training_args,
    eval_dataset=encoded_dataset['validation'],
    compute_metrics=compute_metrics
)

print("Evaluation started.")

results = trainer.evaluate()
print(results)

print("Evaluation finished.")