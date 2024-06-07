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

# Code for reading the bin file
#-----------------------------------------------
def load_word2vec_binary(filename, num_rows=None):
    with open(filename, 'rb') as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        if num_rows is not None:
            vocab_size = min(vocab_size, num_rows)
        binary_len = np.dtype('float32').itemsize * vector_size
        words = []
        vectors = np.empty((vocab_size, vector_size), dtype='float32')
        
        for i in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word).decode('utf-8')
                    break
                if ch != b'\n':  # ignore newline characters
                    word.append(ch)
            words.append(word)
            vectors[i] = np.frombuffer(f.read(binary_len), dtype='float32')
        
    return words, vectors
#-----------------------------------------------

def write_category_associations(file, name_csv, largest_association_num=10000, category="big_tech", category_association_words = ['Google', 'Amazon', 'Facebook', 'Microsoft', 'Apple', 'Nvidia', 'Intel', 'IBM', 'Huawei', 'Samsung', 'Uber', 'Alibaba'], num_rows=1000000, bin_file=False):
    name_csv = f"{category}_associations_{name_csv}.csv"

    if bin_file:
        # Load words and vectors
        words, vectors = load_word2vec_binary(path.join("embeddings", file), num_rows)

        # Create a DataFrame
        embedding_df = pd.DataFrame(vectors, index=words)
        embedding_df.reset_index(inplace=True)
        embedding_df.columns = ['word'] + [f'vec_{i}' for i in range(embedding_df.shape[1] - 1)]
        embedding_df.set_index("word", inplace=True)
    else:
        embedding_df = pd.read_csv(path.join("embeddings", file), sep=' ', header=None, index_col=0, nrows=num_rows, na_values=None, skiprows=1, keep_default_na=False, quoting=csv.QUOTE_NONE)

    #Get mean cosine similarities with Big Tech words
    category_embs = embedding_df.loc[[word for word in category_association_words if word in embedding_df.index]].to_numpy()
    category_normed = category_embs / np.linalg.norm(category_embs,axis=-1,keepdims=True)

    all_embs = embedding_df.to_numpy()
    all_embs_normed = all_embs / np.linalg.norm(all_embs,axis=-1,keepdims=True)

    associations = all_embs_normed @ category_normed.T
    means = np.mean(associations,axis=1)

    #Write dataframe to file
    category_df = pd.DataFrame(means,index=embedding_df.index.tolist(),columns=[f'{category}_es'])
    largest = category_df.nlargest(largest_association_num, f'{category}_es')

    largest.to_csv(path.join("category_associations", category, name_csv))

write_category_associations("crawl-300d-2M.vec", "ft")