from transformers import AutoTokenizer, pipeline, set_seed
import torch

# Set a seed for reproducibility
#set_seed(42)

# Model identifier
model = "yxs33220/llama-2-7b-model-0418-1k"
#model = "llama-2-7b-model-0417_2columns"
# Prompt input from the user
prompt = input("Ask me:")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)

# Automatically select the device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the pipeline for text generation, specifying device and model
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32, # Use float16 on GPU for faster computation
    device=0 if device == "cuda" else -1, # Device index 0 for CUDA, -1 for CPU
)

# Generate sequences
sequences = pipeline(
    f"### Instruction:\n{prompt}\n\n### Response:\n",
    do_sample=True,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=4096,
)

# Print the generated sequences
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
