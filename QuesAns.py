import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load the fine-tuned model
model_path = "/home/lora_finetune/IndicLegalQA_finetune/"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Handle missing pad_token by assigning eos_token as pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the model with the LoRA configuration
lora_config = LoraConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = get_peft_model(model, lora_config)

model.to("cpu")

def generate_answer(prompt, max_tokens=None):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to("cpu") for key, value in inputs.items()}
    
    max_tokens = max_tokens if max_tokens else 100  # Shorten max generation length
    
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


# prompt for generation
example_prompt = """
"case_name": "Bishambhar Prasad vs. M/s Arfat Petrochemicals Pvt. Ltd. & Ors.",
        "judgement_date": "20th April 2023",
        "question": "What did the Supreme Court decide regarding the allocation and use of the land in LIA, Kota?"
Answer:"""

generated_answer = generate_answer(example_prompt)
print(generated_answer)
