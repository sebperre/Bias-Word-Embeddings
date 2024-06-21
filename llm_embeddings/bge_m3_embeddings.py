from FlagEmbedding import BGEM3FlagModel
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
    words_df = pd.read_csv(path.join("embeddings", f'glove.840B.300d.txt'), sep=' ', nrows=num_embeddings, header=None,index_col=0, na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE)
    list_of_words = list(words_df.index)
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)

    embeddings_list = []

    for word in list_of_words:
        embeddings_list.append(model.encode([word])['dense_vecs'][0].tolist())

    df = pd.DataFrame(embeddings_list)
    df.index = list_of_words

    df.to_csv(path.join("llm_10000_embeddings", f"bge_{num_embeddings}_embeddings.csv"), sep=" ", header=False, index=True)

get_embeddings(100000)