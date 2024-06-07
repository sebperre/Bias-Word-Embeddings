import numpy as np
from numpy.core.defchararray import join
from numpy.core.shape_base import _block_slicing
import pandas as pd
from os import path
import os
import csv
from scipy.stats import norm, skew
from matplotlib import pyplot as plt
import seaborn as sns

def SC_WEAT(w, A, B, permutations):
    w_normed = w / np.linalg.norm(w)
    A_normed = A / np.linalg.norm(A,axis=-1,keepdims=True)
    B_normed = B / np.linalg.norm(B,axis=-1,keepdims=True)

    A_associations = w_normed @ A_normed.T
    B_associations = w_normed @ B_normed.T
    joint_associations = np.concatenate((A_associations,B_associations),axis=-1)

    test_statistic = np.mean(A_associations) - np.mean(B_associations)
    effect_size = test_statistic / np.std(joint_associations,ddof=1)

    midpoint = len(A)
    sample_distribution = np.array([np.random.permutation(joint_associations) for _ in range(permutations)])
    sample_associations = np.mean(sample_distribution[:,:midpoint],axis=1) - np.mean(sample_distribution[:,midpoint:],axis=1)
    p_value = 1 - norm.cdf(test_statistic,np.mean(sample_associations),np.std(sample_associations,ddof=1))

    return effect_size, p_value

# Code Necessary for reading the bin file
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

def category_analysis(file, model_compare=["ft", "glove"], category="big_tech", stimuli_pos=['female','woman','girl','sister','she','her','hers','daughter'], stimuli_neg=['male','man','boy','brother','he','him','his','son'], num_rows=1000000, bin_file=False):
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
    
    #Get embeddings
    pos_embeddings, neg_embeddings = embedding_df.loc[stimuli_pos].to_numpy(), embedding_df.loc[stimuli_neg].to_numpy()

    #Read in category associations and take the words that are associated with category in both embeddings
    largest_embedding_1 = pd.read_csv(path.join("category_associations", category,f'{category}_associations_{model_compare[0]}.csv'),index_col=0)
    largest_embedding_2 = pd.read_csv(path.join("category_associations", category, f'{category}_associations_{model_compare[1]}.csv'),index_col=0)

    joint = [i for i in largest_embedding_1.index.tolist() if i in largest_embedding_2.index]

    #Get class associations of category words
    joint_vals = []

    for word in joint:
        joint_emb = embedding_df.loc[word].to_numpy()
        es, p = SC_WEAT(joint_emb,pos_embeddings,neg_embeddings,1000)
        joint_vals.append([es,p])

    joint_arr = np.array(joint_vals)
    cat_df = pd.DataFrame(joint_arr,index=joint,columns=['Effect_Size','P_Value'])
    cat_df.to_csv(path.join("category_analysis", category, f'{category}_weats.csv'))

    #Write category words to file
    words = cat_df.index.tolist()
    with open(path.join("category_analysis", category, f'{category}_words.txt'), "w") as writer:
        writer.write(', '.join(sorted(words,key=str.lower)))

    #Get percentage of category words with minimum gender effect size
    es_list = [0,.2,.5,.8]

    pct_pos, pct_neg = [],[]

    for es in es_list:
        print(es)
        pos_df = cat_df[(cat_df.Effect_Size >= es)]
        pct_pos.append(len(pos_df.index.tolist())/len(cat_df.index.tolist()))

        neg_df = cat_df[(cat_df.Effect_Size <= -es)]
        pct_neg.append(len(neg_df.index.tolist())/len(cat_df.index.tolist()))

    print(pct_pos)
    print(pct_neg)

category_analysis("glove.840B.300d.txt")