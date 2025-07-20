import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
import pandas as pd
import json
import matplotlib.pyplot as plt
import time

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

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['input'], truncation=True, padding="max_length", max_length=512)
    outputs = tokenizer(examples['output'], truncation=True, padding="max_length", max_length=512)
    inputs['labels'] = outputs['input_ids']
    return inputs

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, num_proc=4)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

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

# Define training arguments
training_args = TrainingArguments(
    output_dir="./final_results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    weight_decay=0.01,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=30,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
    no_cuda=True,
    report_to="none",
)

# Track additional metrics
class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_time = []
        self.gradient_norms = []
        self.memory_usage = []

    def on_step_end(self, args, state, control, **kwargs):
        # Measure batch processing time
        start_time = time.time()
        if torch.cuda.is_available():
            memory = torch.cuda.memory_allocated(device)
            self.memory_usage.append(memory / (1024 ** 2))  # Convert to MB
        end_time = time.time()
        self.batch_time.append(end_time - start_time)

        # Measure gradient norm
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)

# Initialize the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save logs
training_logs = trainer.state.log_history
pd.DataFrame(training_logs).to_csv("training_logs.csv", index=False)

# Generate Plots
logs = pd.read_csv("training_logs.csv")

# Training Loss
training_logs = logs[logs['loss'].notnull()]
plt.plot(training_logs['step'], training_logs['loss'], label="Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Over Steps")
plt.legend()
plt.grid()
plt.show()

# Validation Loss
eval_logs = logs[logs['eval_loss'].notnull()]
plt.plot(eval_logs['step'], eval_logs['eval_loss'], label="Validation Loss", color="orange")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Validation Loss Over Steps")
plt.legend()
plt.grid()
plt.show()

# Learning Rate
if "learning_rate" in logs.columns:
    plt.plot(logs['step'], logs['learning_rate'], label="Learning Rate", color="green")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Over Steps")
    plt.legend()
    plt.grid()
    plt.show()

# Gradient Norm
plt.plot(range(len(trainer.gradient_norms)), trainer.gradient_norms, label="Gradient Norm")
plt.xlabel("Step")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm Over Steps")
plt.legend()
plt.grid()
plt.show()

# GPU Memory Usage
if trainer.memory_usage:
    plt.plot(range(len(trainer.memory_usage)), trainer.memory_usage, label="GPU Memory Usage (MB)")
    plt.xlabel("Step")
    plt.ylabel("Memory (MB)")
    plt.title("GPU Memory Usage Over Steps")
    plt.legend()
    plt.grid()
    plt.show()

# Batch Processing Time
plt.plot(range(len(trainer.batch_time)), trainer.batch_time, label="Batch Processing Time (s)")
plt.xlabel("Step")
plt.ylabel("Time (s)")
plt.title("Batch Processing Time Over Steps")
plt.legend()
plt.grid()
plt.show()