import json
import torch
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

def prepare_dataset(json_file_path):
    """
    Prepare the dataset from JSON file for fine-tuning
    """
    # Load dataset
    with open(json_file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Extract data entries
    if isinstance(raw_data, dict) and 'data' in raw_data:
        data = raw_data['data']
    else:
        data = raw_data
    
    # Format each example
    formatted_data = []
    for item in data:
        # Combine prompt and email into a single text
        text = f"PROMPT: {item['prompt']}\nEMAIL: {item['email_body']}"
        formatted_data.append({"text": text})
    
    # Create dataset object
    dataset = Dataset.from_list(formatted_data)
    return dataset

def tokenize_data(examples, tokenizer):
    """
    Tokenize the texts with padding and truncation
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding='max_length',
        max_length=512,
        return_special_tokens_mask=True
    )

def train_model(dataset_path, output_dir="./email_model", batch_size=8):
    """
    Set up and run the fine-tuning process
    """
    # Initialize tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(dataset_path)
    
    # Split dataset
    train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    
    train_data = train_testvalid['train']
    valid_data = test_valid['train']
    test_data = test_valid['test']
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_tokenized = train_data.map(
        lambda x: tokenize_data(x, tokenizer),
        batched=True,
        remove_columns=train_data.column_names
    )
    valid_tokenized = valid_data.map(
        lambda x: tokenize_data(x, tokenizer),
        batched=True,
        remove_columns=valid_data.column_names
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir='./logs',
        save_steps=1000,
        save_total_limit=2,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")
    
    return trainer

if __name__ == "__main__":
    # Path to your dataset
    dataset_path = "email_dataset.json"
    
    # Run training
    trainer = train_model(dataset_path)
    print("Training complete!")
