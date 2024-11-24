import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import pandas as pd
import json

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model and tokenizer configuration
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Handle missing pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
file_path = r"/home/lora_finetune/IndiclegalQA.json"
with open(file_path, 'r') as f:
    dataset = json.load(f)

df = pd.DataFrame(dataset)

# Prepare prompts
def prepare_qa_prompt(row):
    prompt = f"Case: {row['case_name']}\nDate: {row['judgement_date']}\nQuestion: {row['question']}\nAnswer:"
    return prompt, row['answer']

df['input'], df['output'] = zip(*df.apply(prepare_qa_prompt, axis=1))
hf_dataset = Dataset.from_pandas(df[['input', 'output']])

# Tokenize dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['input'], truncation=True, padding="max_length", max_length=512)
    outputs = tokenizer(examples['output'], truncation=True, padding="max_length", max_length=512)
    inputs['labels'] = outputs['input_ids']
    return inputs

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, num_proc=4)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Split dataset
train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load model with CPU offloading and GPU computation
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    offload_folder="./offload", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True    
)

# Apply LoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, config)
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./IndicLegalQA_results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,  
    learning_rate=2e-4,
    weight_decay=0.01,
    num_train_epochs=4,
    logging_dir='./logs',
    logging_steps=30,
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    save_total_limit=2,
    report_to="none",
    fp16=True, 
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train model
trainer.train()

# Save fine-tuned model
trainer.save_model("/home/lora_finetune/IndicLegalQA_finetune")