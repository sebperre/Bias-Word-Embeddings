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

def association_bar_chart(file, axis="Gender", pos_class="Female", neg_class="Male", stimuli_pos=['female','woman','girl','sister','she','her','hers','daughter'], stimuli_neg=['male','man','boy','brother','he','him','his','son']):
    """
    num_rows = 100000

    embedding_df = pd.read_csv(path.join("embeddings", file), sep=' ', header=None, index_col=0, nrows=num_rows, na_values=None, skiprows=0, keep_default_na=False, quoting=csv.QUOTE_NONE)
    pos_embeddings, neg_embeddings = embedding_df.loc[stimuli_pos].to_numpy(), embedding_df.loc[stimuli_neg].to_numpy()

    num_positive = 0
    num_negative = 0

    count_list = []

    for i in range(100000):
        es, p = SC_WEAT(embedding_df.iloc[i], pos_embeddings, neg_embeddings, 1000)
        if es > 0:
            num_positive += 1
        elif es < 0:
            num_negative += 1
        if i == 99 or i == 999 or i == 9999 or i == 99999:
            count_list.append((num_positive / (i+1))*100)
            count_list.append((num_negative / (i+1))*100)
    """
    fast_100k_df = pd.read_csv(path.join("top_100k_words", "ft_100k.csv"), na_values=None, keep_default_na=False)
    glove_100k_df = pd.read_csv(path.join("top_100k_words", "glove_100k.csv"), na_values=None, keep_default_na=False)

    frequency_ceilings = [100, 1000, 10000, 100000]
    #fast_gender_assoc = []
    #glove_gender_assoc = []
    count_list = []

    for ceiling in frequency_ceilings:
        head_fast_df = fast_100k_df.head(ceiling)
        head_glove_df = glove_100k_df.head(ceiling)

        es_glove_df = head_glove_df.loc[head_glove_df['female_effect_size'] >= 0]
        es_quantity = len(es_glove_df.index.tolist())
        es_percentage = (es_quantity / ceiling) * 100
        count_list.append(es_percentage)

        es_glove_df = head_glove_df.loc[head_glove_df['female_effect_size'] <= 0]
        es_quantity = len(es_glove_df.index.tolist())
        es_percentage = (es_quantity / ceiling) * 100
        count_list.append(es_percentage)

        es_fast_df = head_fast_df.loc[head_fast_df['female_effect_size'] >= 0]
        es_quantity = len(es_fast_df.index.tolist())
        es_percentage = (es_quantity / ceiling) * 100
        count_list.append(es_percentage)

        es_fast_df = head_fast_df.loc[head_fast_df['female_effect_size'] <= 0]
        es_quantity = len(es_fast_df.index.tolist())
        es_percentage = (es_quantity / ceiling) * 100
        count_list.append(es_percentage)

    data = {
        'N': ['10^2', '10^2', '10^2', '10^2', 
            '10^3', '10^3', '10^3', '10^3', 
            '10^4', '10^4', '10^4', '10^4', 
            '10^5', '10^5', '10^5', '10^5'],
        'Association': [f'GloVe {pos_class}', f'GloVe {neg_class}', f'fastText {pos_class}', f'fastText {neg_class}',
                        f'GloVe {pos_class}', f'GloVe {neg_class}', f'fastText {pos_class}', f'fastText {neg_class}',
                        f'GloVe {pos_class}', f'GloVe {neg_class}', f'fastText {pos_class}', f'fastText {neg_class}',
                        f'GloVe {pos_class}', f'GloVe {neg_class}', f'fastText {pos_class}', f'fastText {neg_class}'],
        'Percentage': count_list
    }

    # Creating DataFrame
    df = pd.DataFrame(data)

    # Setting the aesthetic style of the plots
    sns.set(style="whitegrid")

    # Creating a bar plot
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='N', y='Percentage', hue='Association', data=df, ci=None)

    # Customizing the plot
    bar_plot.set_xlabel('$N$ Most Frequent Words', fontsize=12, fontstyle='italic')
    bar_plot.set_ylabel(f'% {axis} Association', fontsize=12)
    bar_plot.set_title(f'{axis} Association by Frequency Range', fontsize=14)
    bar_plot.legend(title='')

    # Adjusting the legend
    handles, labels = bar_plot.get_legend_handles_labels()
    bar_plot.legend(handles=handles, labels=labels, title='', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Display the plot
    plt.tight_layout()
    plt.show()

association_bar_chart("glove.840B.300d.txt")