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
wandb.init(project="Llama2_training_0418")
base_model = "meta-llama/Llama-2-7b-hf" #, use_auth_token="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"
new_model = "llama-2-7b-model-0418-1k" #give a new model name
#hf_token="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"
# Dataset
csv_file_path = "/home/cc/dataset/dataset2train_1k_0418.csv"
dataset = load_dataset("csv", split="train", data_files=csv_file_path)
#dataset['train'].to_pandas()
#dataset = load_dataset("yxs33220/llmtestyaman_0405", split="train" , token="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")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, token="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")
tokenizer.pad_token = tokenizer.eos_token #padding to make same length endofsentence, unk-nown 
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
        num_train_epochs=4, #4
        per_device_train_batch_size=4, #10
        gradient_accumulation_steps=4, #1
        gradient_checkpointing= True,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_steps=1,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=10,
        weight_decay=0.01,              # Strength of weight decay
        #report_to="wandb",
        #max_steps=2, # Remove this line for a real fine-tuning
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Start training
trainer.train()

# Save the trained model
trainer.model.save_pretrained(new_model)

'''
 # Run text generation pipeline with our model
prompt = "Given values for dns.count.add_rr = -1, dns.count.answers = -1, dns.count.auth_rr = -1, dns.count.queries = -1, dns.dnskey.algorithm = -1, dns.dnskey.flags = -1, dns.dnskey.flags.key_revoked = -1, dns.dnskey.flags.reserved = -1, dns.dnskey.flags.secure_entry_point = -1, dns.dnskey.flags.zone_key = -1, dns.dnskey.key_id = -1, dns.dnskey.protocol = -1, dns.dnskey.public_key = -1, dns.flags = -1, dns.flags.authenticated = -1, dns.flags.authoritative = -1, dns.flags.checkdisable = -1, dns.flags.conflict = -1, dns.flags.opcode = -1, dns.flags.rcode = -1, dns.flags.recavail = -1, dns.flags.recdesired = -1, dns.flags.response = -1, dns.flags.tentative = -1, dns.flags.truncated = -1, dns.flags.z = -1, dns.id = -1, dns.nsec3.algo = -1, dns.nsec3.flags = -1, dns.nsec3.flags.opt_out = -1, dns.nsec3.hash_length = -1, dns.nsec3.hash_value = -1, dns.nsec3.iterations = -1, dns.nsec3.salt_length = -1, dns.nsec3.salt_value = -1, dns.qry.class = -1, dns.qry.name = -1, dns.qry.qu = -1, dns.qry.type = -1, dns.resp.addr = -1, dns.resp.class = -1, dns.resp.len = -1, dns.resp.name = -1, dns.resp.ns = -1, dns.resp.primaryname = -1, dns.resp.ttl = -1, dns.resp.type = -1, dns.response_to = -1, dns.rrsig.algorithm = -1, dns.rrsig.key_tag = -1, dns.rrsig.labels = -1, dns.rrsig.original_ttl = -1, dns.rrsig.signature = -1, dns.rrsig.signature_expiration = -1, dns.rrsig.signature_inception = -1, dns.rrsig.signers_name = -1, dns.rrsig.type_covered = -1, dns.soa.expire_limit = -1, dns.soa.mininum_ttl = -1, dns.soa.mname = -1, dns.soa.refresh_interval = -1, dns.soa.retry_interval = -1, dns.soa.rname = -1, dns.soa.serial_number = -1, http.accept_encoding = 3222419580, http.cache_control = 263441550, http.connection = 4094582480, http.content_length = -1, http.content_length_header = -1, http.content_type = -1, http.cookie = -1, http.prev_request_in = -1, http.referer = 1864970628, http.request = 1, http.request.method = 4183344954, http.request.version = 1135209249, ip.dsfield = 0, ip.dsfield.dscp = 0, ip.flags = 2, ip.flags.df = 1, ip.flags.mf = 0, ip.frag_offset = 0, ip.fragment = -1, ip.fragment.count = -1, ip.fragments = -1, ip.hdr_len = 20, ip.len = 410, ip.opt.len = -1, ip.opt.ra = -1, ip.opt.type = -1, ip.opt.type.class = -1, ip.opt.type.copy = -1, ip.opt.type.number = -1, ip.proto = 6, ip.reassembled.data = -1, ip.reassembled.length = -1, ip.version = 4, tcp.ack = 1, tcp.analysis = 1, tcp.analysis.ack_lost_segment = -1, tcp.analysis.ack_rtt = -1.0, tcp.analysis.acks_frame = -1, tcp.analysis.bytes_in_flight = 358, tcp.analysis.duplicate_ack = -1, tcp.analysis.duplicate_ack_frame = -1, tcp.analysis.duplicate_ack_num = -1, tcp.analysis.flags = -1, tcp.analysis.lost_segment = -1, tcp.analysis.out_of_order = -1, tcp.analysis.retransmission = -1, tcp.analysis.rto = -1, tcp.analysis.rto_frame = -1, tcp.analysis.window_update = -1, tcp.flags = 24, tcp.flags.ack = 1, tcp.flags.cwr = 0, tcp.flags.fin = 0, tcp.flags.push = 1, tcp.flags.res = 0, tcp.flags.reset = 0, tcp.flags.syn = 0, tcp.flags.urg = 0, tcp.hdr_len = 32, tcp.len = 358, tcp.option_kind = 8, tcp.option_len = 10, tcp.options = 1555135903, tcp.options.mss = -1, tcp.options.mss_val = -1, tcp.options.sack = -1, tcp.options.sack.count = -1, tcp.options.sack_le = -1, tcp.options.sack_perm = -1, tcp.options.sack_re = -1, tcp.options.wscale.multiplier = -1, tcp.options.wscale.shift = -1, tcp.urgent_pointer = -1, tcp.window_size = 502, tcp.window_size_scalefactor = 4294967294, tcp.window_size_value = 502, udp.length = -1, data_source = Test/Packets/Yaman, what is this packet?"
instruction = f"### Instruction:\n{prompt}\n\n### Response:\n"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=4096)
result = pipe(instruction)
print(result[0]['generated_text'][len(instruction):])
'''
# Empty VRAM
del model
#del pipe
del trainer
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

