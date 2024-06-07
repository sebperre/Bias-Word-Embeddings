import pandas as pd
from os import path
import csv

# Load the first CSV file
csv1 = pd.read_csv(path.join("llm_10000_embeddings", "cohere_large_10000_embeddings_1.csv"), sep=" ", quoting=csv.QUOTE_NONE)

# Load the second CSV file
csv2 = pd.read_csv(path.join("llm_10000_embeddings", "cohere_large_10000_embeddings_2.csv"), sep=" ", quoting=csv.QUOTE_NONE)

# Concatenate the two DataFrames
combined_csv = pd.concat([csv1, csv2], ignore_index=False)

# Save the combined result to a new CSV file
combined_csv.to_csv(path.join("llm_10000_embeddings", "cohere_large_10000_embeddings.csv"), index=False, sep=" ")