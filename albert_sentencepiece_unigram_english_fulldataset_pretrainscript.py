# %%
# Import all the necessary libraries
import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from tokenizers import SentencePieceUnigramTokenizer
import multiprocessing
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AlbertConfig, AlbertForMaskedLM

# %%
# Load dataset
bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset('wikipedia', '20220301.en', split='train')

# Clean wiki dataset to keep only text col
wiki = wiki.remove_columns([col for col in wiki.column_names if col != 'text'])

# Compare structure of both datasets, halt if not the same
assert bookcorpus.features.type == wiki.features.type, "Dataset structures are not the same"

# Concatonate both datasets and shuffle
concatenated_datasets = concatenate_datasets([bookcorpus, wiki]).shuffle(seed=114514)
small_test_dataset = concatenated_datasets.select(range(10000)) # select small subset for quick testing


# %%
######## This is used for subsequent code ########
dataset_name = "concatenated_datasets" # Specify save 
dataset = concatenated_datasets # Specify dataset
vocab_size = 30000 # Define vocab size
tokenizer_id = "my_sentencepiece_en" + "_vocab_" + str(vocab_size) + "_[" + dataset_name + "]" # Name the tokenizer
save_path = r'./albert_pretrained_model_final_VocabSize_' + str(vocab_size) + '_dataset_[' + str(tokenizer_id) +']' # define save path
#########################################

# %%
# Inherit from AlbertTokenizerFast
pretrained_tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')  
special_tokens = list(pretrained_tokenizer.all_special_tokens)

print('Special tokens: ' + str(special_tokens))

# Create a generator to dynamically load text data in chunks
def batch_iterator(dataset,batch_size=10000):
    for i in tqdm(range(0, len(dataset), batch_size)):
        yield dataset[i: i + batch_size]["text"]

# %%
# Train tokenizer using bartch iterator
my_albert_tokenizer = pretrained_tokenizer.train_new_from_iterator(
    text_iterator=batch_iterator(dataset=dataset),
    vocab_size=vocab_size,
)

# Save the tokenizer to local disk
my_albert_tokenizer.save_pretrained(f"./{tokenizer_id}")


# %%
# Get CPU count
num_proc = multiprocessing.cpu_count()
print(f"Using {num_proc} CPU cores for tokenization.")

# Define a tokenization function
def tokenize_text(examples, tokenizer=None):
    return tokenizer(
        examples["text"], # Extract 'text' column from dataset
        return_special_tokens_mask=True, # Required for special pretraining task, returns an additional arr for special tokens such as MASK or SEP.
        truncation=True, # Tell the tokenizer to cut off texts longer than max_length
        padding="max_length", # Pad texts to the max length
        max_length=512 # specifies the max length
    )

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

# Tokenize dataset with multiprocessing
tokenized_datasets = dataset.map(
    tokenize_text,
    batched=True,
    remove_columns=['text'],
    num_proc=num_proc,
    fn_kwargs={'tokenizer': tokenizer},
    desc="Tokenizing the dataset",
)

# Shuffle text
tokenized_datasets = tokenized_datasets.shuffle(seed=1919810)
print(f"the dataset contains in total {len(tokenized_datasets)*512} tokens")


# %%
# Define training arguments

######### Change Batch Size Here #########
batch_size = 16
##########################################
training_args = TrainingArguments(
    output_dir=save_path,
    disable_tqdm=False,

    # --- Critical arguments ---
    learning_rate=1e-5,             # A good starting learning rate for pre-training
    num_train_epochs=3,             # Or use max_steps
    per_device_train_batch_size=batch_size, # Adjust based on your GPU/CPU memory
    
    # --- Stability ---
    warmup_steps=1000,              # Number of steps to slowly ramp up the LR
    weight_decay=0.01,              # Standard regularization
    
    # --- Other useful settings ---
    logging_steps=100,              # Log loss every 100 steps
    save_steps=1000,                # Save a checkpoint every 1000 steps
    fp16=torch.cuda.is_available(), # Use mixed precision (Requires a GPU)
)

# Override the config to match new tokenizer
config = AlbertConfig(
    vocab_size=tokenizer.vocab_size
)

# Load model architecture
model = AlbertForMaskedLM(config=config)

# Define data collator for MLM task
# Data collator will dynamically mask tokens during training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# %%
# Clear Mem before training
import torch, gc
gc.collect()
torch.cuda.empty_cache()

# Define Trainer
# A trainer is responsible for the training loop
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator
)

# Check for GPU availability
if torch.cuda.is_available():
    print(f"GPU is available and ready!")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: PyTorch cannot find a CUDA-enabled GPU.")
    print("Training will run on the CPU (very slow).")

# %%
# Start training
trainer.train()

# Save the trained model to disk
trainer.save_model(save_path)

# Save the tokenizer too
tokenizer.save_pretrained(str(save_path) + "/tokenizer/")

# Save the config and training arguments
config.save_pretrained(save_path)
training_args.save(save_path)
