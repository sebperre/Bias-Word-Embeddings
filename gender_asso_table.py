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

num_rows = 100000

file = "glove_100k.csv"

source_df = pd.read_csv(path.join("top_100k_words", file), na_values=None, keep_default_na=False)

frequency_ceilings = [100, 1000, 10000, 100000]
effect_size_floors = [0, 0.2, 0.5, 0.8]

female_count = []
male_count = []
percent_female_count = []
percent_male_count = []

for ceiling in frequency_ceilings:
    head_df = source_df.head(ceiling)

    temp_female = []
    temp_male = []

    temp_percent_female = []
    temp_percent_male = []
    for es in effect_size_floors:
        temp_female_df = head_df.loc[head_df["female_effect_size"] >= es]
        temp_quantity = len(temp_female_df.index.tolist())
        temp_female.append(temp_quantity)

        temp_male_df = head_df.loc[head_df["female_effect_size"] <= -es]
        temp_quantity = len(temp_male_df.index.tolist())
        temp_male.append(temp_quantity)

        temp_percent_female.append(round((temp_female[-1]/(temp_female[-1] + temp_male[-1])) * 100))
        temp_percent_male.append(round((temp_male[-1]/(temp_female[-1] + temp_male[-1])) * 100))
    female_count.append(temp_female)
    male_count.append(temp_male)

    print(f"Female count for n={ceiling}, {female_count}")
    print(f"Male count for n={ceiling}, {male_count}")

    percent_female_count.append(temp_percent_female)
    percent_male_count.append(temp_percent_male)

# Data
data = {
    'N Most Frequent Words': ['N = 100', 'N = 1,000', 'N = 10,000', 'N = 100,000'],
    'd > 0.00 Female': [f'{female_count[0][0]} ({percent_female_count[0][0]}%)', f'{female_count[1][0]} ({percent_female_count[1][0]}%)', f'{female_count[2][0]} ({percent_female_count[2][0]}%)', f'{female_count[3][0]} ({percent_female_count[3][0]}%)'],
    'd > 0.00 Male': [f'{male_count[0][0]} ({percent_male_count[0][0]}%)', f'{male_count[1][0]} ({percent_male_count[1][0]}%)', f'{male_count[2][0]} ({percent_male_count[2][0]}%)', f'{male_count[3][0]} ({percent_male_count[3][0]}%)'],
    'd > 0.20 Female': [f'{female_count[0][1]} ({percent_female_count[0][1]}%)', f'{female_count[1][1]} ({percent_female_count[1][1]}%)', f'{female_count[2][1]} ({percent_female_count[2][1]}%)', f'{female_count[3][1]} ({percent_female_count[3][1]}%)'],
    'd > 0.20 Male': [f'{male_count[0][1]} ({percent_male_count[0][1]}%)', f'{male_count[1][1]} ({percent_male_count[1][1]}%)', f'{male_count[2][1]} ({percent_male_count[2][1]}%)', f'{male_count[3][1]} ({percent_male_count[3][1]}%)'],
    'd > 0.50 Female': [f'{female_count[0][2]} ({percent_female_count[0][2]}%)', f'{female_count[1][2]} ({percent_female_count[1][2]}%)', f'{female_count[2][2]} ({percent_female_count[2][2]}%)', f'{female_count[3][2]} ({percent_female_count[3][2]}%)'],
    'd > 0.50 Male': [f'{male_count[0][2]} ({percent_male_count[0][2]}%)', f'{male_count[1][2]} ({percent_male_count[1][2]}%)', f'{male_count[2][2]} ({percent_male_count[2][2]}%)', f'{male_count[3][2]} ({percent_male_count[3][2]}%)'],
    'd > 0.80 Female': [f'{female_count[0][3]} ({percent_female_count[0][3]}%)', f'{female_count[1][3]} ({percent_female_count[1][3]}%)', f'{female_count[2][3]} ({percent_female_count[2][3]}%)', f'{female_count[3][3]} ({percent_female_count[3][3]}%)'],
    'd > 0.80 Male': [f'{male_count[0][3]} ({percent_male_count[0][3]}%)', f'{male_count[1][3]} ({percent_male_count[1][3]}%)', f'{male_count[2][3]} ({percent_male_count[2][3]}%)', f'{male_count[3][3]} ({percent_male_count[3][3]}%)'],
}

# Creating DataFrame
df = pd.DataFrame(data)

# Writing DataFrame to CSV
csv_file_path = 'gender_association.csv'
df.to_csv(path.join("charts_and_tables", csv_file_path), index=False)