from openai import OpenAI
import os
import numpy as np
from numpy.core.defchararray import join
from numpy.core.shape_base import _block_slicing
import pandas as pd
from os import path
import csv
from scipy.stats import norm, skew
from matplotlib import pyplot as plt
import seaborn as sns
import struct
from io import BytesIO

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def openai_embed(query: str, model="text-embedding-3-large"):
   query = query.replace("\n", " ")
   response = openai_client.embeddings.create(input=[query], model=model)
   embedding = response.data[0].embedding
   return embedding

def get_embeddings(num_embeddings, model="base"):
    words_df = pd.read_csv(path.join("embeddings", f'glove.840B.300d.txt'), sep=' ', nrows=num_embeddings, header=None,index_col=0, na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE)
    list_of_words = list(words_df.index)

    embeddings_list = []

    for word in list_of_words:
        embeddings_list.append(openai_embed(word))

    df = pd.DataFrame(embeddings_list)
    df.index = list_of_words

    df.to_csv(path.join("llm_10000_embeddings", f"openai_{num_embeddings}_embeddings.csv"), sep=" ", header=False, index=True)

get_embeddings(100)