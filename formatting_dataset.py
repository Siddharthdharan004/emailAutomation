from datasets import Dataset
import json

# Load the dataset
with open("improved_organizational_email_dataset.json", "r") as f:
    data = json.load(f)

# Combine prompt and email_body into one text sample
formatted_data = [
    {"text": f"Prompt: {item['prompt']}\nEmail: {item['email_body']}"} for item in data
]

# Create a Dataset object
dataset = Dataset.from_list(formatted_data)

# Split into train and validation sets
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

print("Sample data:", train_dataset[0]["text"])
