ALBERT Pretraining from Scratch
Custom English ALBERT model trained on BookCorpus + Wikipedia

Python 3.9+
PyTorch 2.0+
Transformers (Hugging Face)
Status: Active

Overview

This project demonstrates how to pretrain an ALBERT (A Lite BERT) model from scratch using:

BookCorpus and Wikipedia (English) datasets

A custom SentencePiece tokenizer

The Hugging Face Transformers and Datasets libraries

You can reproduce a simplified version of the original ALBERT pretraining pipeline with this script.

Features

Trains a custom SentencePiece tokenizer (configurable vocab size)

Efficient multiprocessing tokenization

Full Masked Language Modeling (MLM) training loop with Hugging Face Trainer

Saves model, tokenizer, config, and training arguments for reuse

Supports GPU + FP16 training if available

Setup

Requirements:
pip install torch datasets transformers tokenizers tqdm

Recommended:

At least 50 GB disk space

CUDA-enabled GPU for efficient training

Usage

Clone the repository:
git clone https://github.com/yourusername/albert-pretraining.git

cd albert-pretraining

Run the training script:
python train_albert.py

Tip: Use a small subset (small_test_dataset) for quick debugging.

Key Parameters

vocab_size: Vocabulary size for tokenizer (default 30000)
batch_size: Training batch size per device (default 16)
max_length: Maximum token length per sequence (default 512)
num_train_epochs: Number of training epochs (default 3)
learning_rate: Learning rate (default 1e-5)
mlm_probability: Masking probability (default 0.15)

Script Breakdown

Dataset Loading

Loads BookCorpus and Wikipedia datasets.

Cleans and concatenates them into one dataset.

Tokenizer Training

Trains a new SentencePiece unigram tokenizer from iterator batches.

Saves tokenizer to ./my_sentencepiece_en_vocab_30000_[concatenated_datasets]/

Dataset Tokenization

Uses all CPU cores for fast parallel tokenization.

Removes the original text column and outputs tokenized tensors.

Model and Training Setup

Creates an ALBERT configuration with vocab size = tokenizer.vocab_size.

Uses AlbertForMaskedLM for masked language modeling.

Defines training arguments (epochs, batch size, learning rate, etc.).

Training Loop

Uses Hugging Face Trainer for training and logging.

Supports GPU + mixed precision (FP16) if available.

Output Structure

After training, the model and artifacts are saved as:

albert_pretrained_model_final_VocabSize_30000_dataset_[my_sentencepiece_en_vocab_30000_[concatenated_datasets]]/
│
├── config.json
├── pytorch_model.bin
├── tokenizer/
├── training_args.bin
└── vocab.txt (or .model depending on tokenizer)

You can reload the model later with:
from transformers import AlbertForMaskedLM, AlbertTokenizer
model = AlbertForMaskedLM.from_pretrained("./path/to/model")
tokenizer = AlbertTokenizer.from_pretrained("./path/to/model/tokenizer")

Tips

Reduce batch_size if you encounter CUDA out-of-memory errors.

Experiment with vocab_size for better language coverage.

Test with a small dataset before full-scale pretraining.

Example Results

Metric: Loss (MLM training loss) → ~2.4 after 3 epochs
Tokens processed: ~5B (depends on dataset size and training duration)