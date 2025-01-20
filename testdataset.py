import json

# Read first few entries
with open('email_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    # Print the first email
    print("First email in dataset:")
    print(data['data'][0]['prompt'])
    print("\n")
    print(data['data'][0]['email_body'])
    print("\nTotal number of emails:", len(data['data']))
