import numpy as np
import pandas as pd
from scipy.stats import norm
from os import path
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

#Attribute Words
female_stimuli = ['female','woman','girl','sister','she','her','daughter'] #'hers'
male_stimuli = ['male','man','boy','brother','he','him','his','son']

#Skip first row when reading FT, not when reading GloVe

embedding_df = pd.read_csv(path.join("llm_10000_embeddings", f'bge_10000_embeddings.csv'), sep=' ', nrows=10000, header=None,index_col=0, na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE)
print('BGE loaded')

female_embeddings, male_embeddings = embedding_df.loc[female_stimuli].to_numpy(), embedding_df.loc[male_stimuli].to_numpy()
embedding_targets = embedding_df.index.tolist()[:10000]

#NRC-VAD Dataframe
vad_df = pd.read_table(path.join("lexicons", f'NRC-VAD-Lexicon.txt'),sep='\t',index_col=0, na_values=None, keep_default_na=False)
vad_words = vad_df.index.tolist()
vad_words = [word for word in vad_words if word in embedding_df.index]

gender_biases, p_values = [],[]

#VAD WEATs - GloVe embedding
bias_array = np.array([SC_WEAT(embedding_df.loc[word].to_numpy(),female_embeddings,male_embeddings,PERMUTATIONS) for word in [str(i) for i in vad_words]])
bias_df = pd.DataFrame(bias_array,index=vad_words,columns=['female_effect_size','female_p_value'])
#bias_df.to_csv(path.join("gender_association_collector", "vad", f'glove_vad_words.csv'))
print('BGE VAD')

#10k WEATS at a time - 100k most frequent words - GloVe embedding
targets = embedding_targets[0:STEP]
bias_array = np.array([SC_WEAT(embedding_df.loc[word].to_numpy(),female_embeddings,male_embeddings,PERMUTATIONS) for word in targets])
bias_df = pd.DataFrame(bias_array,index=targets,columns=['female_effect_size','female_p_value'])
bias_df.to_csv(path.join("llm_gender_association_collector", f'BGE_10k.csv'))

print('BGE 10k')

##########################

#Read in FastText embedding
embedding_ft = pd.read_csv(path.join("llm_10000_embeddings", f'cohere_large_10000_embeddings.csv'),sep=' ', nrows=10000, header=None,index_col=0, na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE)
print('Cohere loaded')

female_embeddings, male_embeddings = embedding_ft.loc[female_stimuli].to_numpy(), embedding_ft.loc[male_stimuli].to_numpy()
embedding_targets = embedding_ft.index.tolist()[:10000]

#Only VAD words in embedding
vad_words = vad_df.index.tolist()
vad_words = [word for word in vad_words if word in embedding_ft.index]

bias_array = np.array([SC_WEAT(embedding_ft.loc[word].to_numpy(),female_embeddings,male_embeddings,PERMUTATIONS) for word in [str(i) for i in vad_words]])
bias_ft = pd.DataFrame(bias_array,index=vad_words,columns=['female_effect_size','female_p_value'])
#bias_ft.to_csv(path.join("gender_association_collector", "vad", f'ft_vad_words.csv'))
print('Cohere VAD')

targets = embedding_targets[0:STEP]
bias_array = np.array([SC_WEAT(embedding_ft.loc[word].to_numpy(),female_embeddings,male_embeddings,PERMUTATIONS) for word in targets])
bias_ft = pd.DataFrame(bias_array,index=targets,columns=['female_effect_size','female_p_value'])
bias_ft.to_csv(path.join("llm_gender_association_collector", f'cohere_10k.csv'))

print('Cohere 10k')

##########################

embedding_df = pd.read_csv(path.join("llm_10000_embeddings", f'e5_base_10000_embeddings.csv'), sep=' ', nrows=10000, header=None,index_col=0, na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE)
print('e5 loaded')

female_embeddings, male_embeddings = embedding_df.loc[female_stimuli].to_numpy(), embedding_df.loc[male_stimuli].to_numpy()
embedding_targets = embedding_df.index.tolist()[:10000]

#NRC-VAD Dataframe
vad_df = pd.read_table(path.join("lexicons", f'NRC-VAD-Lexicon.txt'),sep='\t',index_col=0, na_values=None, keep_default_na=False)
vad_words = vad_df.index.tolist()
vad_words = [word for word in vad_words if word in embedding_df.index]

gender_biases, p_values = [],[]

#VAD WEATs - GloVe embedding
bias_array = np.array([SC_WEAT(embedding_df.loc[word].to_numpy(),female_embeddings,male_embeddings,PERMUTATIONS) for word in [str(i) for i in vad_words]])
bias_df = pd.DataFrame(bias_array,index=vad_words,columns=['female_effect_size','female_p_value'])
#bias_df.to_csv(path.join("gender_association_collector", "vad", f'glove_vad_words.csv'))
print('e5 VAD')

#10k WEATS at a time - 100k most frequent words - GloVe embedding
targets = embedding_targets[0:STEP]
bias_array = np.array([SC_WEAT(embedding_df.loc[word].to_numpy(),female_embeddings,male_embeddings,PERMUTATIONS) for word in targets])
bias_df = pd.DataFrame(bias_array,index=targets,columns=['female_effect_size','female_p_value'])
bias_df.to_csv(path.join("llm_gender_association_collector", f'e5_10k.csv'))

print('e5 10k')

##########################

embedding_df = pd.read_csv(path.join("llm_10000_embeddings", f'openai_3_large_10000_embeddings.csv'), sep=' ', nrows=10000, header=None,index_col=0, na_values=None, keep_default_na=False, quoting=csv.QUOTE_NONE)
print('openai loaded')

female_embeddings, male_embeddings = embedding_df.loc[female_stimuli].to_numpy(), embedding_df.loc[male_stimuli].to_numpy()
embedding_targets = embedding_df.index.tolist()[:10000]

#NRC-VAD Dataframe
vad_df = pd.read_table(path.join("lexicons", f'NRC-VAD-Lexicon.txt'),sep='\t',index_col=0, na_values=None, keep_default_na=False)
vad_words = vad_df.index.tolist()
vad_words = [word for word in vad_words if word in embedding_df.index]

gender_biases, p_values = [],[]

#VAD WEATs - GloVe embedding
bias_array = np.array([SC_WEAT(embedding_df.loc[word].to_numpy(),female_embeddings,male_embeddings,PERMUTATIONS) for word in [str(i) for i in vad_words]])
bias_df = pd.DataFrame(bias_array,index=vad_words,columns=['female_effect_size','female_p_value'])
#bias_df.to_csv(path.join("gender_association_collector", "vad", f'glove_vad_words.csv'))
print('openai VAD')

#10k WEATS at a time - 100k most frequent words - GloVe embedding
targets = embedding_targets[0:STEP]
bias_array = np.array([SC_WEAT(embedding_df.loc[word].to_numpy(),female_embeddings,male_embeddings,PERMUTATIONS) for word in targets])
bias_df = pd.DataFrame(bias_array,index=targets,columns=['female_effect_size','female_p_value'])
bias_df.to_csv(path.join("llm_gender_association_collector", f'openai_10k.csv'))

print('openai 10k')


