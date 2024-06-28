import numpy as np
import pandas as pd
from scipy.stats import norm
from os import path
import os
import csv
import codecs
import re

#SC-WEAT function
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

#This code obtains the SC-WEAT association for the top 100,000 most frequent words in each embedding, and for the words in NRC-VAD Lexicon

#Constants
STEP = 10000
PERMUTATIONS = 10000
GLOVE_DIR = f'glove.840B.300d'

def generate_associations(axis="gender", pos_name="female", pos_stimuli=['female','woman','girl','sister','she','her','daughter', 'hers'], neg_stimuli=['male','man','boy','brother','he','him','his','son']):
    llms = ["bge", "cohere", "e5", "openai"]
    if not os.path.exists(path.join("llm_association_collector", f"llm_{axis}_association_collector")):
        print("Created Directories")
        os.makedirs(path.join("llm_association_collector", f"llm_{axis}_association_collector"))
        for llm in llms:
            os.makedirs(path.join("llm_association_collector", f"llm_{axis}_association_collector", llm))

    for llm in llms:
        embedding_df = pd.read_csv(path.join("llm_10000_embeddings", f'{llm}_100000_embeddings.csv'), sep=' ', nrows=100000, header=None,index_col=0, na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE)
        print(f'{llm} loaded')

        pos_embeddings, neg_embeddings = embedding_df.loc[pos_stimuli].to_numpy(), embedding_df.loc[neg_stimuli].to_numpy()
        embedding_targets = embedding_df.index.tolist()[:100000]

        for i in range(10):
            targets = embedding_targets[i*STEP:(i+1)*STEP]
            bias_array = np.array([SC_WEAT(embedding_df.loc[word].to_numpy(),pos_embeddings,neg_embeddings,PERMUTATIONS) for word in targets])  
            bias_df = pd.DataFrame(bias_array,index=targets,columns=[f'{pos_name}_effect_size',f'{pos_name}_p_value'])
            bias_df.to_csv(path.join("llm_association_collector", f"llm_{axis}_association_collector", llm, f'{llm}_100k_{i}.csv'))

        print(f'{llm} 100k')

        # Concatenate and save 10k-word association dataframes
        concat_ = []
        for i in range(10):
            df = pd.read_csv(path.join("llm_association_collector", f"llm_{axis}_association_collector", llm, f'{llm}_100k_{i}.csv'),names=['word',f'{pos_name}_effect_size','p_value'],skiprows=1,index_col='word', na_values=None, keep_default_na=False)
            concat_.append(df)

        full_df = pd.concat(concat_,axis=0)
        full_df.to_csv(path.join("llm_association_collector", f"llm_{axis}_association_collector", f'{llm}_100k.csv'))


generate_associations(axis="race", pos_name="caucasian", pos_stimuli=["white", "caucasian", "european", "nordic"], neg_stimuli=["black", "african", "dark", "ebony"])

