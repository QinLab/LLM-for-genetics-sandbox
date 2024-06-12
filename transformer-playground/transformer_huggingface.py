from datasets import load_dataset
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
# tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

# tokenized_dataset = tokenizer(
#     raw_datasets["train"]["sentence1"],
#     raw_datasets["train"]["sentence2"],
#     padding=True,
#     truncation=True,
# )

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# predictions = trainer.predict(tokenized_datasets["validation"])
# print(predictions.predictions.shape, predictions.label_ids.shape)
# preds = np.argmax(predictions.predictions, axis=-1)

# metric = evaluate.load("glue", "mrpc")
# metric.compute(predictions=preds, references=predictions.label_ids)