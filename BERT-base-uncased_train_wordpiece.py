# %% [markdown]
# 1. Prepare Dataset

######## Important Note ########
# Change the vocab size and dataset according to your need
# Make sure the vocab size in both tokenizer training and model config are the same
################################

# %%
# Import dataset before loading
from datasets import concatenate_datasets, load_dataset

# Custon save location
# save_location = r"D:\LM\BERT_pretrain_practice\load_dataset"

# Load bookcorpus dataset
bookcorpus = load_dataset("bookcorpus", split="train")
# Save to disk at custom location
# bookcorpus.save_to_disk(f"{save_location}/bookcorpus")

# %%
# Load wikipedia dataset
# If encountering issue 'Dataset scripts are no longer supported, but found'
#   There might be an issue with the newest huggingface_hub or datasets release
#   Try to downgrade to previous version: pip install datasets==2.16.0
wiki = load_dataset('wikipedia', '20220301.en', split='train')


# Clean wiki dataset to keep only text col
wiki = wiki.remove_columns([col for col in wiki.column_names if col != 'text'])

# %%
# Compare structure of both datasets, halt if not the same
assert bookcorpus.features.type == wiki.features.type, "Dataset structures are not the same"

# %%
# Concatonate both datasets
contatonated_dataset = concatenate_datasets([bookcorpus, wiki])

# Reduced size dataset
small_test_dataset = contatonated_dataset.shuffle(seed=114514).select(range(50000))

# %% [markdown]
# 2. Train Tokenizer

# %%
# Next, we train a tokenizer on the dateset
from tqdm import tqdm
from transformers import BertTokenizerFast

# Tokenizer id here
#   It uses bert-base-uncased and Habana Gaudi platform optimizations
tokenizer_id = "bert-base-uncased"

# %%
# Create a generator to dynamically load text data in chunks
def batch_iterator(dataset,batch_size=10000):
    for i in tqdm(range(0, len(dataset), batch_size)):
        yield dataset[i: i + batch_size]["text"]

# Create a tokenizer from the pretrained tokenizer to re-use exisiting special tokens
pretrained_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# %%
# Specify vocab size, reduce for lighter training
#   Original BERT uses 30,522

####### Change vocab size here #######
volcab_size = 6000

# Start the training of tokenizer 
#   Inherit from the pretrained tokenizer to re-use special tokens
#   The training method provided by the transformers library, using WordPiece algorithm
my_bert_tokenizer = pretrained_tokenizer.train_new_from_iterator(
    text_iterator=batch_iterator(small_test_dataset),
    vocab_size=volcab_size
)

# Save the tokenizer to local disk
my_bert_tokenizer.save_pretrained(f"./{tokenizer_id}" + "_volcab_size_" + str(volcab_size))

# %% [markdown]
# 3. Preprocess Data

# %%
# Load libraries
from transformers import AutoTokenizer
import multiprocessing

# %%
# Load the newly trained tokenizer from disk
tokenizer = AutoTokenizer.from_pretrained(f"./{tokenizer_id}" + "_volcab_size_" + str(volcab_size))
# Define number of processes for multiprocessing
num_proc = multiprocessing.cpu_count()
print(f"using {num_proc} processes for tokenization")
print(f"The max length of the tokenizer is {tokenizer.model_max_length}")

# %%
# Define a tokenization function
def tokenize_text(examples, tokenizer=None):
    return tokenizer(
        examples["text"], # Extract 'text' column from dataset
        return_special_tokens_mask=True, # Required for special pretraining task, returns an additional arr for special tokens such as MASK or SEP.
        truncation=True, # Tell the tokenizer to cut off texts longer than max_length
        max_length=tokenizer.model_max_length, # specifies the max length
    )

# %%
# # Use one sample to test the tokenization function
# sample_size = 2
# examples = small_test_dataset.select(range(sample_size))
# # Test tokenization function
# test_tokenized_text_example = examples.map(
#     tokenize_text,
#     batched=True,
#     remove_columns=['text'],
#     num_proc=num_proc,
#     fn_kwargs={'tokenizer': tokenizer},
#     desc="Tokenizing the dataset",
# )
# test_tokenized_text_example.features

# %%
# Tokenize dataset with multiprocessing

######### Change dataset here #########
tokenized_datasets = small_test_dataset.map(
    tokenize_text,
    batched=True,
    remove_columns=['text'],
    num_proc=num_proc,
    fn_kwargs={'tokenizer': tokenizer},
    desc="Tokenizing the dataset",
)
tokenized_datasets.features

# %%
# Shuffle text
tokenized_datasets = tokenized_datasets.shuffle(seed=34)
print(f"the dataset contains in total {len(tokenized_datasets)*tokenizer.model_max_length} tokens")

# %% [markdown]
# 4. Train the model

# %%
# import libraries for training
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, BertForMaskedLM, BertConfig

# %%
# Define training arguments
training_args = TrainingArguments(
    output_dir='./bert_pretrained_model',
)

# Load configuration for BERT model
config = BertConfig.from_pretrained('bert-base-uncased')

# Modify the config according to need
# If using the smaller vocab size, the embedding size must be changed accordingly
# e.g. volcab_size = 6000 for our test case
config.vocab_size = volcab_size

# Load model architecture
# Set config=config if using modified config
# BertForMaskedLM is used for masked language modeling task
# It loads the model with random weights if no pretrained weights are specified
# To load pretrained weights, use model = BertForMaskedLM(config=config)
model = BertForMaskedLM(config=config)

# Define data collator for MLM task
# Data collator will dynamically mask tokens during training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# %%
# Define Trainer
# A trainer is responsible for the training loop
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator
)

# %%
# Start training
trainer.train()

# Save the trained model to disk
trainer.save_model("./bert_pretrained_model_final")

# Save the tokenizer too
tokenizer.save_pretrained("./bert_pretrained_model_final")


