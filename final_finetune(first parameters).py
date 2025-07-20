import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
import pandas as pd
import json

# Login to Hugging Face Hub
login()  # You'll be prompted to enter your Hugging Face token

# Check GPU availability for dataset processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for dataset processing: {device}")

# Model and tokenizer configuration
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Handle missing pad_token by assigning eos_token as pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set eos_token as pad_token

# Load your dataset (using the provided JSON dataset)
file_path = r"/home/apulkit/embedding/all_combined_10k.json"

# Load and prepare the dataset
with open(file_path, 'r') as f:
    dataset = json.load(f)

df = pd.DataFrame(dataset)

# Function to create the prompt (input) and the expected output
def prepare_qa_prompt(row):
    # Prepare the prompt by concatenating context (case name, date, and question) as input
    prompt = f"Case: {row['case_name']}\nDate: {row['judgement_date']}\nQuestion: {row['question']}\nAnswer:"
    return prompt, row['answer']

# Apply preparation and create a DataFrame for the prompts and answers
df['input'], df['output'] = zip(*df.apply(prepare_qa_prompt, axis=1))

# Convert to Hugging Face Dataset format
hf_dataset = Dataset.from_pandas(df[['input', 'output']])

# Tokenize the dataset on the GPU (if available)
def tokenize_function(examples):
    # Tokenize input and output separately for causal LM fine-tuning
    inputs = tokenizer(examples['input'], truncation=True, padding="max_length", max_length=512)
    outputs = tokenizer(examples['output'], truncation=True, padding="max_length", max_length=512)  # Match input/output length

    # Prepare the labels for causal LM (use padding same as input)
    inputs['labels'] = outputs['input_ids']
    
    return inputs

# Tokenize the dataset in batches on the GPU (if available)
tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, num_proc=4)  # Use multiple processes for speed
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Move the dataset to GPU if available
if torch.cuda.is_available():
    tokenized_dataset = tokenized_dataset.with_format("torch", device="cuda")

# Split the dataset into training and validation
train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load the model on CPU
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map={"": "cpu"}  # Explicitly place model on CPU
)

# Prepare for LoRA fine-tuning
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

# Define training arguments for CPU training
training_args = TrainingArguments(
    output_dir="./final_results",
    per_device_train_batch_size=4,  # Batch size for CPU
    gradient_accumulation_steps=16,  # Increase gradient accumulation to save memory
    learning_rate=2e-4,
    weight_decay=0.01,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=30,
    report_to="none",  # Disable reporting to wandb or Hugging Face
    fp16=False,  # Disable mixed precision as it is not supported on CPU
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
    no_cuda=True,  # Explicitly disable CUDA for model training
)

# Initialize the Trainer on CPU
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune the model on CPU
trainer.train()

# Save the fine-tuned model
trainer.save_model("/home/apulkit/embedding/final_lora_finetuned_llama")
