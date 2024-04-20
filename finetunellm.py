# pip install -q -U transformers datasets accelerate peft trl bitsandbytes wandb

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

import wandb
import os 
wandb.login(key='434e686675c31aa5fd943573df5d3c5559cecaf4')
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#wandb.login(key=434e686675c31aa5fd943573df5d3c5559cecaf4)
# Model
wandb.init(project="Llama2_training_500data")
base_model = "meta-llama/Llama-2-7b-hf" #, use_auth_token="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"
new_model = "llama-2-7b-model-0415500" #give a new model name
#hf_token="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"
# Dataset
csv_file_path = "/home/cc/dataset/newdataset0415.csv"
dataset = load_dataset("csv", split="train", data_files=csv_file_path)
#dataset = load_dataset("yxs33220/llmtestyaman_0405", split="train" , token="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")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, token="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")
tokenizer.pad_token = tokenizer.unk_token #padding to make same length
tokenizer.padding_side = "right"

# Quantization configuration QLORA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# Load base moodel
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map={"": 0}
)

# Cast the layernorm in fp32, make output embedding layer require grads, add the upcasting of the lmhead to fp32
model = prepare_model_for_kbit_training(model)

# Set training arguments
training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_steps=1,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=10,
        weight_decay=0.01,              # Strength of weight decay
        report_to="wandb",
        #max_steps=2, # Remove this line for a real fine-tuning
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="instruction",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Empty GPU cache before starting the training
#torch.cuda.empty_cache()

# Start training
trainer.train()

# Empty GPU cache after training has completed
#torch.cuda.empty_cache()

# Save the trained model
trainer.model.save_pretrained(new_model)


 # Run text generation pipeline with our model
prompt = "what is goldeneye packet?,"
instruction = f"### Instruction:\n{prompt}\n\n### Response:\n"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=4096)
result = pipe(instruction)
print(result[0]['generated_text'][len(instruction):])

# Empty VRAM
del model
del pipe
del trainer
import gc
torch.cuda.empty_cache()

# Reload model in FP16 and merge it with LoRA weights
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_auth_token="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")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.push_to_hub(new_model, use_temp_dir=False, token="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")
tokenizer.push_to_hub(new_model, use_temp_dir=False, token="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")