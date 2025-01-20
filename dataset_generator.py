import json
import random
from datetime import datetime, timedelta
import faker
import numpy as np

class EmailDatasetGenerator:
    def __init__(self):
        self.fake = faker.Faker()
        self.formal_templates = {
            'business': [
                "Subject: {project} Update - Q{quarter} {year}\n\nDear {name},\n\n{content}\n\nBest regards,\n{sender}",
                "Subject: Meeting Invitation - {topic}\n\nDear {name},\n\n{content}\n\nKind regards,\n{sender}",
                "Subject: Proposal for {project}\n\nDear {name},\n\n{content}\n\nSincerely,\n{sender}"
            ],
            'client': [
                "Subject: Welcome to {company}!\n\nDear {name},\n\n{content}\n\nBest wishes,\n{sender}",
                "Subject: Your Recent Inquiry about {product}\n\nDear {name},\n\n{content}\n\nBest regards,\n{sender}"
            ]
        }
        
        self.informal_templates = {
            'personal': [
                "Subject: {event} plans!\n\nHey {name}!\n\n{content}\n\nCheers,\n{sender}",
                "Subject: Quick question about {topic}\n\nHi {name},\n\n{content}\n\nThanks!\n{sender}"
            ],
            'team': [
                "Subject: {event} next week\n\nHey team!\n\n{content}\n\nThanks,\n{sender}",
                "Subject: Quick update on {project}\n\nHey everyone,\n\n{content}\n\nBest,\n{sender}"
            ]
        }

    def generate_business_content(self):
        paragraphs = random.randint(2, 5)
        content = []
        for _ in range(paragraphs):
            sentences = random.randint(2, 6)
            paragraph = ' '.join(self.fake.sentences(sentences))
            content.append(paragraph)
        return '\n\n'.join(content)

    def generate_informal_content(self):
        paragraphs = random.randint(1, 4)
        content = []
        for _ in range(paragraphs):
            sentences = random.randint(1, 4)
            paragraph = ' '.join(self.fake.sentences(sentences))
            content = content + [paragraph]
        return '\n\n'.join(content)

    def generate_prompt(self, category, context):
        prompts = {
            'business': [
                f"Write a professional email about {context}",
                f"Draft a formal email regarding {context}",
                f"Compose a business update about {context}"
            ],
            'personal': [
                f"Write a casual email about {context}",
                f"Send a friendly note regarding {context}",
                f"Draft an informal message about {context}"
            ]
        }
        return random.choice(prompts[category])

    def generate_dataset(self, size_mb=200):
        dataset = []
        estimated_entry_size = 1024  # Approximate size in bytes for each entry
        num_entries = (size_mb * 1024 * 1024) // estimated_entry_size
        
        for _ in range(num_entries):
            is_formal = random.random() > 0.4  # 60% formal, 40% informal
            
            if is_formal:
                category = random.choice(list(self.formal_templates.keys()))
                template = random.choice(self.formal_templates[category])
                content = self.generate_business_content()
            else:
                category = random.choice(list(self.informal_templates.keys()))
                template = random.choice(self.informal_templates[category])
                content = self.generate_informal_content()
            
            context = self.fake.catch_phrase()
            email = template.format(
                name=self.fake.name(),
                sender=self.fake.name(),
                company=self.fake.company(),
                project=self.fake.bs(),
                product=self.fake.catch_phrase(),
                event=self.fake.word(),
                topic=context,
                quarter=random.randint(1, 4),
                year=random.randint(2023, 2025),
                content=content
            )
            
            entry = {
                "prompt": self.generate_prompt(
                    'business' if is_formal else 'personal',
                    context
                ),
                "email_body": email,
                "metadata": {
                    "type": "formal" if is_formal else "informal",
                    "category": category,
                    "length": len(email),
                    "timestamp": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
                }
            }
            dataset.append(entry)
        
        return dataset

def save_dataset(filename="email_dataset.json", size_mb=200):
    generator = EmailDatasetGenerator()
    dataset = generator.generate_dataset(size_mb)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({"data": dataset, "metadata": {
            "size": len(dataset),
            "generated_at": datetime.now().isoformat(),
            "version": "1.0"
        }}, f, indent=2)
    return len(dataset)
