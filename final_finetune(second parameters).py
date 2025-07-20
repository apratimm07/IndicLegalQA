import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
import pandas as pd
import json
import os

# Set environment variable to manage memory fragmentation in PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Login to Hugging Face Hub
login()  # You'll be prompted to enter your Hugging Face token

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for dataset processing: {device}")

# Model and tokenizer configuration
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Handle missing pad_token by assigning eos_token as pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load your dataset (using the provided JSON dataset)
file_path = r"/home/apulkit/embedding/all_combined_10k.json"

with open(file_path, 'r') as f:
    dataset = json.load(f)

df = pd.DataFrame(dataset)

# Function to create the prompt (input) and the expected output
def prepare_qa_prompt(row):
    prompt = f"Case: {row['case_name']}\nDate: {row['judgement_date']}\nQuestion: {row['question']}\nAnswer:"
    return prompt, row['answer']

df['input'], df['output'] = zip(*df.apply(prepare_qa_prompt, axis=1))

# Convert to Hugging Face Dataset format
hf_dataset = Dataset.from_pandas(df[['input', 'output']])

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['input'], truncation=True, padding="max_length", max_length=512)
    outputs = tokenizer(examples['output'], truncation=True, padding="max_length", max_length=512)

    # Prepare labels
    inputs['labels'] = outputs['input_ids']
    
    return inputs

# Tokenize the dataset in batches, using multiple processes for speed
tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, num_proc=4)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Split the dataset into training and validation
train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # Enable 8-bit loading to save memory
    device_map="auto",  # Automatically allocate layers across available devices
)

# LoRA configuration for fine-tuning
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
)

# Apply LoRA to the model
model = get_peft_model(model, config)

# Define training arguments, adapted for GPU usage
training_args = TrainingArguments(
    output_dir="./last_results",
    per_device_train_batch_size=2,  # Reduced batch size to avoid memory fragmentation
    gradient_accumulation_steps=16,  # Increased gradient accumulation to reduce memory load
    learning_rate=2e-4,
    weight_decay=0.01,
    num_train_epochs=4,
    logging_dir='./logs',
    logging_steps=30,
    report_to="none",  # Disable reporting to wandb or Hugging Face
    fp16=True,  # Enable mixed precision training to save memory
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
)

# Initialize the Trainer on GPU if available
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("/home/apulkit/embedding/last_lora_finetuned_llama")

# Empty GPU cache periodically to avoid memory fragmentation
torch.cuda.empty_cache()
