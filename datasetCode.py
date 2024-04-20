# python3.9 -m venv llmtest
# virtual environment source /home/cc/llmtest/bin/activate


# Install libraries
#!pip install -q datasets transformers sentence_transformers faiss-gpu
# rm -rf ~/.cache/huggingface/datasets/*  to clear the cache
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset

# Load the dataset
csv_file_path = "/home/cc/dataset/newdataset0415.csv"
dataset = load_dataset("csv", data_files=csv_file_path)
#dataset = load_dataset("/home/cc/dataset/llama_training_data04-04.csv") #, use_auth_token=True)
# Read as pandas DataFrame
dataset['train'].to_pandas()
#this will convert raw text into tokens
print(dataset['train'])

from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import torch
# Load model directly

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token="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")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", use_auth_token="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")

#creating a tokens; refer to the datasets header for more info
instruction_token_counts = [len(tokenizer.tokenize(example["instruction"])) for example in dataset['train']]
output_token_counts = [len(tokenizer.tokenize(example["output"])) for example in dataset['train']]
combined_token_counts = [instruction + output for instruction, output in zip(instruction_token_counts, output_token_counts)]
#combined_token_counts; comment afteruse

# Helper function to plot the distributions
def plot_distribution(token_counts, title):  #This Python function, plot_distribution, is designed to create a histogram that visualizes the distribution of token counts in a dataset.
    sns.set_style("whitegrid") #This line sets the style of the plot to "whitegrid," which adds a grid to the plot background for better visualization
    plt.figure(figsize=(15, 6))
    plt.hist(token_counts, bins=50, color='#3498db', edgecolor='black')
    plt.title(title, fontsize=16)
    plt.xlabel("Number of tokens", fontsize=14)
    plt.ylabel("Number of examples", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

# Plot the distribution of token counts
plot_distribution(instruction_token_counts, "Distribution of token counts for instruction only")
plot_distribution(output_token_counts, "Distribution of token counts for output only")
plot_distribution(combined_token_counts, "Distribution of token counts for combined instruction + output")

valid_indices = [i for i, count in enumerate(combined_token_counts) if count <= 2048] #changing default token length size to 2048 from 4096
print(len(valid_indices))
print(len(dataset['train']) - len(valid_indices))

dataset['train'] = dataset['train'].select(valid_indices) #do not execute twice

# Get token counts for valid rows
token_counts = [combined_token_counts[i] for i in valid_indices]

plot_distribution(token_counts, "New distribution of token counts for combined instruction + output")

#embedding model
from sentence_transformers import SentenceTransformer
import faiss
from datasets import Dataset, DatasetDict
from tqdm.autonotebook import tqdm
import numpy as np

def deduplicate_dataset(dataset: Dataset, model: str, threshold: float): #name of the embedding model
    sentence_model = SentenceTransformer(model)
    outputs = [example["output"] for example in dataset['train']]

    print("Converting text to embeddings...")
    embeddings = sentence_model.encode(outputs, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    if normalized_embeddings.ndim != 2:
        raise ValueError("Normalized embeddings are not a 2D array.")
    if np.isnan(normalized_embeddings).any():
        raise ValueError("Normalized embeddings contain NaN values.")

    index.add(normalized_embeddings)

    print("Filtering out near-duplicates...")
    D, I = index.search(normalized_embeddings, k=2)
    to_keep = []

    for i in tqdm(range(len(embeddings)), desc="Filtering"):
        # If the second closest vector (D[i, 1]) has cosine similarity above the threshold
        if D[i, 1] >= threshold:
            # Check if either the current item or its nearest neighbor is already in the to_keep list
            nearest_neighbor = I[i, 1]
            if i not in to_keep and nearest_neighbor not in to_keep:
                # If not, add the current item to the list
                to_keep.append(i)
        else:
            # If the similarity is below the threshold, always keep the current item
            to_keep.append(i)

    dataset = dataset['train'].select(to_keep)
    return DatasetDict({"train": dataset})

deduped_dataset = deduplicate_dataset(dataset, "thenlper/gte-large", 0.95) #name of the embedding model; 0.95 is a threashold; depends on the model we use

print(f"Number of samples in the original dataset: {len(dataset['train'])}")
print(f"Number of samples in the deduped dataset: {len(deduped_dataset['train'])}")
print(f"Number of samples that were removed: {len(dataset['train']) - len(deduped_dataset['train'])}")

# Get the top k rows with the most tokens
def get_top_k_rows(dataset, token_counts, k):
    # Sort by descending token count and get top k indices
    sorted_indices = sorted(range(len(token_counts)), key=lambda i: token_counts[i], reverse=True)
    top_k_indices = sorted_indices[:k]

    # Extract top k rows
    top_k_data = {
        "instruction": [dataset['train'][i]["instruction"] for i in top_k_indices],
        "output": [dataset['train'][i]["output"] for i in top_k_indices]
    }

    return Dataset.from_dict(top_k_data)

# Get token counts
instruction_token_counts = [len(tokenizer.tokenize(example["instruction"])) for example in deduped_dataset['train']]
output_token_counts = [len(tokenizer.tokenize(example["output"])) for example in deduped_dataset['train']]
combined_token_counts = [instruction + output for instruction, output in zip(instruction_token_counts, output_token_counts)]

k = 200000  # You can adjust this value as needed
top_k_dataset = get_top_k_rows(deduped_dataset, combined_token_counts, k)

# Save these rows in a Dataset object with a 'train' split
dataset = DatasetDict({"train": top_k_dataset})

instruction_token_counts = [len(tokenizer.tokenize(example["instruction"])) for example in dataset['train']]
output_token_counts = [len(tokenizer.tokenize(example["output"])) for example in dataset['train']]
combined_token_counts = [instruction + output for instruction, output in zip(instruction_token_counts, output_token_counts)]

# Plot the distribution of token counts
plot_distribution(instruction_token_counts, "Distribution of token counts for instruction only")
plot_distribution(output_token_counts, "Distribution of token counts for output only")
plot_distribution(combined_token_counts, "Distribution of token counts for combined instruction + output")


# Read as pandas DataFrame
dataset['train'].to_pandas()

#creating chat template
def chat_template(example):
    example["instruction"] = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}\n"
    return example

dataset = dataset.map(chat_template)
train_df= dataset['train'].to_pandas()

train_df.to_csv("/home/cc/dataset/newaksyam0406", index=False)

# Optional: push to Hugging Face Hub
#dataset.push_to_hub("yxs33220/llmtestyaman_0405", token="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")
'''