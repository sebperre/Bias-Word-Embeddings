import cohere
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

def get_embeddings(num_embeddings):
   COHERE_API_KEY = os.environ.get("COHERE_PRODUCTION_API_KEY")
   co = cohere.Client(api_key=COHERE_API_KEY) 

   words_df = pd.read_csv(path.join("embeddings", f'glove.840B.300d.txt'), sep=' ', nrows=num_embeddings, header=None,index_col=0, na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE)
   list_of_words = list(words_df.index)

   chunk_size = 1000

   num_chunks = len(list_of_words) // chunk_size

   embeddings_list = []

   for i in range(num_chunks):
      success = False

      while not success:
         try:
            response = co.embed(texts=list_of_words[chunk_size*i:chunk_size*i+chunk_size],input_type='classification', embedding_types=['float'], model='embed-multilingual-v3.0')  
            embeddings = response.embeddings.float # All text embeddings
            success = True
         except:
            print("Failure")
            success = False
      for embedding in embeddings:
         embeddings_list.append(embedding)
      print(f"Chunk {i} Completed out of {num_chunks}")

   if len(list_of_words) % chunk_size != 0:
      response = co.embed(texts=list_of_words[chunk_size*num_chunks:],input_type='classification', embedding_types=['float'], model='embed-multilingual-v3.0')  
      embeddings = response.embeddings.float # All text embeddings
      for embedding in embeddings:
         embeddings_list.append(embedding)
      print("Last Chunk Completed")

   df = pd.DataFrame(embeddings_list)
   df.index = list_of_words

   df.to_csv(path.join("llm_10000_embeddings", f"cohere_{num_embeddings}_embeddings.csv"), sep=" ", header=False, index=True)

get_embeddings(100000)
