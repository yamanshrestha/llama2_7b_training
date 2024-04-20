
import pandas as pd

# Load your dataset
df = pd.read_csv('/home/cc/dataset/dataset_1k_0418.csv')

def row_to_text(row):
    # Formatting each feature:value pair, with $ signs around the values
    features_text = ', '.join([f'${key}: {value}$' for key, value in row.items() if key != 'label'])
    # Extract the label for the current row to use in the explanation
    label_description = row['label']
    # Construct a sentence for the instruction including all features
    #concept = f"{features_text}"
    #output = f"This is categorized as a {label_description} packet."
    text = f"### Instruction:\nFor the given packet values, {features_text}, what is this packet?\n### Response:\nThis is categorized as a {label_description} packet."
    return text #concept, output, 
# Apply the function to each row in the DataFrame to create instruction and output columns
df[['text']] = pd.DataFrame(df.apply(row_to_text, axis=1).tolist(), index=df.index) #,"output"

# Specify the output file path (use a raw string or double backslashes to avoid escape sequence errors)
output_path = r'/home/cc/dataset/dataset2train_1k_0418.csv'

# Save the 'instruction' and 'output' columns to a new CSV file
df[['text']].to_csv(output_path, index=False) #,"output"

print(f"Dataset has been successfully saved to {output_path}")

