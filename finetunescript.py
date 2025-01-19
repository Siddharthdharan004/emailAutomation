from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import json

# Load your dataset
def load_custom_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    prompts = [entry['prompt'] for entry in data]
    email_bodies = [entry['email_body'] for entry in data]
    return {"prompt": prompts, "email_body": email_bodies}

# Use the dataset path
dataset_path = r"C:\Users\NP\Desktop\project1\improved_organizational_email_dataset.json"
dataset = load_custom_dataset(dataset_path)

# Prepare the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))  # Resize embeddings for the new token

# Tokenize the dataset
def preprocess_function(examples):
    inputs = ["Prompt: " + prompt + "\nResponse: " + response for prompt, response in zip(examples['prompt'], examples['email_body'])]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    # Use the input IDs as labels for causal language modeling
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Convert dataset to Hugging Face Dataset
raw_dataset = Dataset.from_dict(dataset)
tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_email_model",
    evaluation_strategy="no",  # Disable evaluation since no eval_dataset is provided
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,  # Use mixed precision if supported
    push_to_hub=False,  # Set to True if pushing to Hugging Face Hub
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_email_model")
tokenizer.save_pretrained("./fine_tuned_email_model")
