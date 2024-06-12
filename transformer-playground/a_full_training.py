from datasets import load_dataset
import evaluate
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, get_scheduler

#load dataset
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Prepare for training
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

#check the batch_size:
for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})
#----------------finish data processing------------------------

#start  instantiating our model:
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)  # inja hanooz train nashode :)

#start writting training loop
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(   #the learning rate scheduler used by default is just a linear decay from the maximum value (5e-5) to 0
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,  #This usually means that you use a very low learning rate for a set number of training steps (warmup steps). 
                         #After your warmup steps you use your "regular" learning rate or learning rate scheduler. You can also gradually 
                         # increase your learning rate over the number of warmup steps.
                         # As far as I know, this has the benefit of slowly starting to tune things like attention mechanisms in your network.
    num_training_steps=num_training_steps,
)
print(num_training_steps)

#The training loop
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

##we are now ready to start training
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}   #tokenizer injast
        outputs = model(**batch) # va model ham injast
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

#The evaluation loop
metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()