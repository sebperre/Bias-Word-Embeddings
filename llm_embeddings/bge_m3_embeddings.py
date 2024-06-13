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


def bge_m3_embed(query: str):
    # Can add "use_fp16=True" to speed up predictions
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
    embeddings = model.encode([query])['dense_vecs'][0]
    return embeddings

# Example usage (1024 dimensions)
embeddings = bge_m3_embed("This is a text I want to embed")
print(embeddings)

def get_embeddings(num_embeddings):
    pd.read_csv(path.join("embeddings", "glove.850B.300d.txt"))