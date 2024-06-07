import numpy as np
import pandas as pd
from os import path

file = "glove_100k.csv"

#Read in file of associations for top 100k words
source_df = pd.read_csv(path.join("top_100k_words", file), na_values=None, keep_default_na=False)

#Female vs. Male ratio by frequency range, effect size range
frequency_ceilings = [100,1000,10000,100000]
effect_size_floors = [0,.2,.5,.8]

es_list = []

for ceiling in frequency_ceilings:
    head_df = source_df.head(ceiling)
    ceiling_counts = [ceiling]

    for es in effect_size_floors:
        es_df = head_df.loc[head_df['female_effect_size'] >= es]
        es_quantity = len(es_df.index.tolist())
        ceiling_counts.append(es_quantity)

    for es in effect_size_floors:
        es_df = head_df.loc[head_df['female_effect_size'] <= -es]
        es_quantity = len(es_df.index.tolist())
        ceiling_counts.append(es_quantity)

    es_list.append(ceiling_counts)

es_arr = np.array(es_list)
cols = ['num_words']+[f'female_{str(i)}' for i in effect_size_floors]+[f'male_{str(i)}' for i in effect_size_floors]
es_df = pd.DataFrame(es_arr,columns=cols)
es_df.to_csv(path.join("effect_sizes", "gender", "glove_100k_female_male_effect_sizes.csv"))