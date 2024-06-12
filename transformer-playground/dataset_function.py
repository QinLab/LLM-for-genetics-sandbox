import html
from datasets import load_dataset
import time
from transformers import AutoTokenizer

#reading files of train and test dataset
data_files = {"train": "/home/qxy699/data_llm/drugsComTrain_raw.tsv", "test": "/home/qxy699/data_llm/drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

#asserting if we have id as many as raws in datasets
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))

if len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0")):
    drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)

#-----------------------------------EXAM:-----------------------------------------   
#Try it out! Use the Dataset.unique() function to find the number of unique drugs and conditions in the training and test sets.
print(f'Number of unique drugs in train dataset: {len(drug_dataset["train"].unique("drugName"))}')
print(f'Number of unique drugs in test dataset: {len(drug_dataset["test"].unique("drugName"))}')

print(f'Number of unique conditions in train dataset: {len(drug_dataset["train"].unique("condition"))}')
print(f'Number of unique conditions in test dataset: {len(drug_dataset["test"].unique("condition"))}')
#-----------------------------------Finish----------------------------------------- 

#filter None entry in dataset in column condition
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

#make all condition normalize by applying map function
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


drug_dataset.map(lowercase_condition)

#Creating new columns
#counting number of words in each review and add the length of review as a coulmn to each row:
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

drug_dataset = drug_dataset.map(compute_review_length)

#drug_dataset["train"].sort("review_length")[:3] # in ascending order

#use the Dataset.filter() function to remove reviews that contain fewer than 30 words.
drug_dataset_copy = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset_copy.num_rows)

#-----------------------------------EXAM:-----------------------------------------  
# Try it out! Use the Dataset.sort() function to inspect the reviews with the largest numbers of words.
drug_descending = drug_dataset["train"].sort("review_length")[-1:]
# print(f'{drug_descending = }')
#-----------------------------------Finish----------------------------------------- 

'''
If you’re running the second code (new_drug_dataset) in a notebook, you’ll see that this command executes way faster than the previous one. 
And it’s not because our reviews have already been HTML-unescaped — if you re-execute the instruction from the previous section 
(without batched=True), it will take the same amount of time as before. This is because list comprehensions are usually faster than executing
 the same code in a for loop, and we also gain some performance by accessing lots of elements at the same time instead of one by one.
'''
# drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})
new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True)

#The map() method’s superpowers
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)

start = time.time()
tokenized_dataset = drug_dataset.map(tokenize_function, batched=True)
end = time.time()
print(f"Time taken: {(end-start)*10**3:.03f}ms")
#-----------------------------------EXAM:----------------------------------------- 
#Try it out! Execute the same instruction with and without batched=True, then try it with a slow tokenizer 
# (add use_fast=False in the AutoTokenizer.from_pretrained() method) so you can see what numbers you get on your hardware.
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)

# start = time.time()
# tokenized_dataset = drug_dataset.map(tokenize_function, batched=True)
# end = time.time()
# print(f"Time taken: {(end-start)*10**3:.03f}ms")

#To enable multiprocessing, use the num_proc argument and specify the number of processes to use in your call to Dataset.map():
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)


def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)

#-------------------------NEW---------------------------
#Let’s have a look at how it works! Here we will tokenize our examples and truncate them to a maximum length of 128, 
# but we will ask the tokenizer to return all the chunks of the texts instead of just the first one. This can be done 
# with return_overflowing_tokens=True:
tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)

def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )

#example: https://huggingface.co/learn/nlp-course/chapter5/3?fw=pt#:~:text=result%20%3D%20tokenize_and_split(drug_dataset%5B%22train%22%5D%5B0%5D)%0A%5Blen(inp)%20for%20inp%20in%20result%5B%22input_ids%22%5D%5D

################tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)  
#warning:  here those 1,000 examples gave 1,463 new features, resulting in a shape error and cause a mismatch in 
# the lengths of one of the columns, one being of length 1,463 and the other of length 1,000.
#The problem is that we’re trying to mix two different datasets of different sizes: the drug_dataset columns will 
# have a certain number of examples (the 1,000 in our error), but the tokenized_dataset we are building will have more 
# (the 1,463 in the error message; it is more than 1,000 because we are tokenizing long reviews into more than one example by 
# using return_overflowing_tokens=True). so we need to either remove the columns from the old dataset or make them the same 
# size as they are in the new dataset. We can do the former with the remove_columns argument:

# tokenized_dataset = drug_dataset.map(
#     tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
# )

#We mentioned that we can also deal with the mismatched length problem by making the old columns the same size as the new ones. 
#To do this, we will need the overflow_to_sample_mapping field the tokenizer returns when we set return_overflowing_tokens=True

def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping") #[0, 0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8, 9, 9, 10, 11, 12, 12, 13, 14, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20,...]
    # print(sample_map)
    for key, values in examples.items():
        # print(key)
        # print(values)
        result[key] = [values[i] for i in sample_map]
    return result

tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
print(tokenized_dataset) 
#YAY!We get the same number of training features as before, but here we’ve kept all the old fields.