from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from tokenizers import Tokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer


import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
import math
from datasets import load_dataset

# Set a configuration for our RoBERTa model
config = RobertaConfig(vocab_size=50000)

# Initialize the model from a configuration without pretrained weights
model = RobertaForMaskedLM(config=config)
print('Num parameters: ', model.num_parameters())

print(f"\n\nLoading Tokenizer....\n\n")
# Create the tokenizer from a trained one
tokenizer = Tokenizer.from_file("Robert/config.json")

def tokenize_function(examples):
    return tokenizer.encode(examples["text"])

file_paths = [str(x) for x in Path("nepali-text").glob("**/*.txt")]

print(f"\n\nLoading Dataset....\n\n")
# Load all the text files for training the model
dataset = load_dataset('text', data_files=file_paths, cache_dir="cache")
dataset = dataset.map(tokenize_function, batched=True, num_proc=16, remove_columns=["text"])

# Define the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)


default_args = {
    "output_dir": "Robert",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}

# Define the training arguments
training_args = TrainingArguments(
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    fp16=True,
    save_steps=10_000,
    save_total_limit=1,
    **default_args,
    do_eval=False,
    evaluation_strategy="no",
)

model.resize_token_embeddings(len(tokenizer))
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset['train'],
)


print(f"\n\nTraining Started....\n\n")
# Train the model
trainer.train()
trainer.save_model("Robert")

# # eval_results = trainer.evaluate()
# # print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# fill_mask = pipeline(
#     "fill-mask",
#     model="Robert",
#     tokenizer="Robert"
# )

# fill_mask("हामीले यसलाई कसरी <mask> गर्न सक्छ?")
# # The test text: Round neck sweater with long sleeves
# fill_mask("तपाईंलाई कस्तो <mask> चाहिएको छ?")
