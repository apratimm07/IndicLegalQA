import json
import re

# Load the dataset
file_path = r'C:\Users\Apratim\OneDrive\Desktop\WORK\Legal QA System\dataset_civil_criminal\all_combined_10k.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize counters and storage for each type
question_types = {
    "Factual": [],
    "Interpretive": [],
    "Procedural": [],
    "Comparative": [],
    "Contextual": [],
    "Clarifying": [],
    "Predictive": [],
    "Definitional": [],
    "Causal": [],
    "Hypothetical": [],
    "Implication": [],
    "Legal Principle": [],
    "Descriptive": []
}

# Define keywords and patterns for each question type
patterns = {
    "Factual": r"\b(who|what|when|where|which|how many|how much)\b",
    "Interpretive": r"\b(why|explain|interpret|analyze)\b",
    "Procedural": r"\b(how|steps|procedure|method|process)\b",
    "Comparative": r"\b(compare|difference between|distinguish|versus|contrast)\b",
    "Contextual": r"\b(context|background|situation|circumstance)\b",
    "Clarifying": r"\b(clarify|elaborate|more details|in detail)\b",
    "Predictive": r"\b(will|would|expect|future|predict)\b",
    "Definitional": r"\b(define|meaning of|what does.*mean|explanation of)\b",
    "Causal": r"\b(cause|origin|reason for|what led to)\b",
    "Hypothetical": r"\b(if|suppose|assume|hypothetically)\b",
    "Implication": r"\b(implication|impact|consequence|effect on)\b",
    "Legal Principle": r"\b(principle|doctrine|legal basis|underlying law)\b",
    "Descriptive": r"\b(describe|explain|outline|overview)\b"
}

# Classify questions based on patterns
for item in data:
    question_text = item['question'].lower()
    classified = False
    
    for q_type, pattern in patterns.items():
        if re.search(pattern, question_text):
            question_types[q_type].append(item)
            classified = True
            break  # Assign only one category per question

# Display counts for each category
for q_type, questions in question_types.items():
    print(f"{q_type}: {len(questions)} questions")

# Optionally, save categorized questions to a new JSON file
output_path = r'C:\Users\Apratim\OneDrive\Desktop\WORK\Legal QA System\dataset_civil_criminal\questions.json'
with open(output_path, 'w') as output_file:
    json.dump(question_types, output_file, indent=4)

print(f"Categorized questions saved to {output_path}")
