import json
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load tokenizer and model
model_path = "/home/apulkit/embedding/lora_finetune/IndicLegalQA_finetune/"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = get_peft_model(model, lora_config)

# Ensure the model is on CPU
model.to("cpu")

# Function to generate an answer from a prompt
def generate_answer(prompt, max_tokens=None):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to("cpu") for key, value in inputs.items()}
    
    max_tokens = max_tokens if max_tokens else 120  # Shorten max generation length
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,  # Avoid repeating trigrams
        repetition_penalty=1.2,  # Penalize repetition
        do_sample=True,  # Disable sampling for deterministic output
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove repeated sentences
    unique_sentences = list(dict.fromkeys(answer.split(". ")))
    return ". ".join(unique_sentences)

# Load the dataset
input_file = "/home/apulkit/embedding/lora_finetune/eval_data.json"
output_file = "/home/apulkit/embedding/lora_finetune/output_data.json"

with open(input_file, "r") as f:
    dataset = json.load(f)

results = []

# Iterate through each record in the dataset
for i, record in enumerate(dataset):
    case_name = record["case_name"]
    judgement_date = record["judgement_date"]
    question = record["question"]

    # Log the data being evaluated
    print(f"Processing record {i+1}/{len(dataset)}: case_name={case_name}, judgement_date={judgement_date}, question={question}")
    
    # Construct the prompt
    prompt = f"\"case_name\": \"{case_name}\",\n        \"judgement_date\": \"{judgement_date}\",\n        \"question\": \"{question}\"\nAnswer:"

    # Generate answer
    answer = generate_answer(prompt)
    
    # Print the generated answer
    print(f"Generated Answer: {answer}\n")
    
    # Append to results
    results.append({
        "case_name": case_name,
        "judgement_date": judgement_date,
        "question": question,
        "answer": answer
    })

# Save the results to a file
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Generated answers saved to {output_file}")
