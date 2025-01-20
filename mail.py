from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_email(prompt, model_path="./final_model"):
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    # Prepare input
    input_text = f"PROMPT: {prompt}\nEMAIL:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Generate
    output = model.generate(
        input_ids,
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7
    )
    
    # Decode and return
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "Write a professional email about project update"
generated_email = generate_email(prompt)
print(generated_email)
