from transformers import pipeline

# Load your fine-tuned email generation model
model_tag = "./fine_tuned_email_model"  # Path to the fine-tuned model
generator = pipeline('text-generation', model=model_tag)

# Define your email prompt
prompt = """
Subject: Follow-up on the Project Deadline

Dear Team,
I hope this email finds you well. I'm writing to discuss our upcoming deadlines.
"""

# Generate the email
result = generator(prompt, max_length=150, num_return_sequences=1, do_sample=True)

# Print the generated email
print("Generated Email:")
print(result[0]['generated_text'])
