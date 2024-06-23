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

def table_association(file, csv_file_name, pos_class_name="female", neg_class_name="male"):
    es_df = pd.read_csv(path.join("llm_gender_association_collector", file), na_values=None, keep_default_na=False)

    n_list = [100, 1000, 10000, 100000]
    es_list = [0, 0.2, 0.5, 0.8]

    pos_count = []
    neg_count = []
    percent_pos_count = []
    percent_neg_count = []

    for number in n_list:
        temp_es_df = es_df.head(number)

        temp_pos = []
        temp_neg = []

        temp_percent_pos = []
        temp_percent_neg = []
        for es in es_list:
            temp_pos.append(len(temp_es_df[temp_es_df[f"{pos_class_name}_effect_size"] > es]))
            temp_neg.append(len(temp_es_df[temp_es_df[f"{pos_class_name}_effect_size"] < es]))

            temp_percent_pos.append(round((temp_pos[-1]/(temp_pos[-1] + temp_neg[-1])) * 100))
            temp_percent_neg.append(round((temp_neg[-1]/(temp_pos[-1] + temp_neg[-1])) * 100))
        pos_count.append(temp_pos)
        neg_count.append(temp_neg)

        percent_pos_count.append(temp_percent_pos)
        percent_neg_count.append(temp_percent_neg)

    # Data
    data = {
        'N Most Frequent Words': ['N = 100', 'N = 1,000', 'N = 10,000', 'N = 100,000'],
        f'd > 0.00 {pos_class_name}': [f'{pos_count[0][0]} ({percent_pos_count[0][0]}%)', f'{pos_count[1][0]} ({percent_pos_count[1][0]}%)', f'{pos_count[2][0]} ({percent_pos_count[2][0]}%)', f'{pos_count[3][0]} ({percent_pos_count[3][0]}%)'],
        f'd > 0.00 {neg_class_name}': [f'{neg_count[0][0]} ({percent_neg_count[0][0]}%)', f'{neg_count[1][0]} ({percent_neg_count[1][0]}%)', f'{neg_count[2][0]} ({percent_neg_count[2][0]}%)', f'{neg_count[3][0]} ({percent_neg_count[3][0]}%)'],
        f'd > 0.20 {pos_class_name}': [f'{pos_count[0][1]} ({percent_pos_count[0][1]}%)', f'{pos_count[1][1]} ({percent_pos_count[1][1]}%)', f'{pos_count[2][1]} ({percent_pos_count[2][1]}%)', f'{pos_count[3][1]} ({percent_pos_count[3][1]}%)'],
        f'd > 0.20 {neg_class_name}': [f'{neg_count[0][1]} ({percent_neg_count[0][1]}%)', f'{neg_count[1][1]} ({percent_neg_count[1][1]}%)', f'{neg_count[2][1]} ({percent_neg_count[2][1]}%)', f'{neg_count[3][1]} ({percent_neg_count[3][1]}%)'],
        f'd > 0.50 {pos_class_name}': [f'{pos_count[0][2]} ({percent_pos_count[0][2]}%)', f'{pos_count[1][2]} ({percent_pos_count[1][2]}%)', f'{pos_count[2][2]} ({percent_pos_count[2][2]}%)', f'{pos_count[3][2]} ({percent_pos_count[3][2]}%)'],
        f'd > 0.50 {neg_class_name}': [f'{neg_count[0][2]} ({percent_neg_count[0][2]}%)', f'{neg_count[1][2]} ({percent_neg_count[1][2]}%)', f'{neg_count[2][2]} ({percent_neg_count[2][2]}%)', f'{neg_count[3][2]} ({percent_neg_count[3][2]}%)'],
        f'd > 0.80 {pos_class_name}': [f'{pos_count[0][3]} ({percent_pos_count[0][3]}%)', f'{pos_count[1][3]} ({percent_pos_count[1][3]}%)', f'{pos_count[2][3]} ({percent_pos_count[2][3]}%)', f'{pos_count[3][3]} ({percent_pos_count[3][3]}%)'],
        f'd > 0.80 {neg_class_name}': [f'{neg_count[0][3]} ({percent_neg_count[0][3]}%)', f'{neg_count[1][3]} ({percent_neg_count[1][3]}%)', f'{neg_count[2][3]} ({percent_neg_count[2][3]}%)', f'{neg_count[3][3]} ({percent_neg_count[3][3]}%)'],
    }

    # Creating DataFrame
    df = pd.DataFrame(data)

    # Writing DataFrame to CSV
    df.to_csv(path.join("charts_and_tables", csv_file_name), index=False)

table_association("BGE_100k.csv", "gender_association_BGE.csv")
table_association("cohere_100k.csv", "gender_association_cohere.csv")
table_association("e5_100k.csv", "gender_association_e5.csv")
table_association("openai_100k.csv", "gender_association_openai.csv")