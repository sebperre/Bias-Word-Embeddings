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

def calc_es_percentage(count_list, ceiling, df):
    es_df = df.loc[df['female_effect_size'] >= 0]
    es_quantity = len(es_df.index.tolist())
    es_percentage = (es_quantity / ceiling) * 100
    count_list.append(es_percentage)

    es_df = df.loc[df['female_effect_size'] <= 0]
    es_quantity = len(es_df.index.tolist())
    es_percentage = (es_quantity / ceiling) * 100
    count_list.append(es_percentage)


def association_bar_chart_indiv(file, llm, axis="Gender", pos_class="Female", neg_class="Male", stimuli_pos=['female','woman','girl','sister','she','her','hers','daughter'], stimuli_neg=['male','man','boy','brother','he','him','his','son']):
    file_df = pd.read_csv(path.join("llm_gender_association_collector", file), na_values=None, keep_default_na=False)

    frequency_ceilings = [100, 1000, 10000]

    count_list = []

    for ceiling in frequency_ceilings:
        head_file_df = file_df.head(ceiling)

        calc_es_percentage(count_list, ceiling, head_file_df)

    data = {
        'N': ['10^2', '10^2',
            '10^3', '10^3',
            '10^4', '10^4'],
            #'10^5', '10^5', '10^5', '10^5', '10^5', '10^5', '10^5', '10^5'],
        'Association': [f'{llm} {pos_class}', f'{llm} {neg_class}',
                        f'{llm} {pos_class}', f'{llm} {neg_class}',
                        f'{llm} {pos_class}', f'{llm} {neg_class}'],
                        #f'BGE {pos_class}', f'BGE {neg_class}', f'Cohere {pos_class}', f'Cohere {neg_class}', f'BGE {pos_class}', f'BGE {neg_class}', f'Cohere {pos_class}', f'Cohere {neg_class}'],
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
    bar_plot.set_title(f'{axis} Association by Frequency Range ({llm} Model)', fontsize=14)
    bar_plot.legend(title='')

    # Adjusting the legend
    handles, labels = bar_plot.get_legend_handles_labels()
    bar_plot.legend(handles=handles, labels=labels, title='', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Display the plot
    plt.tight_layout()
    plt.show()

def all_association_bar_chart(axis="Gender", pos_class="Female", neg_class="Male", stimuli_pos=['female','woman','girl','sister','she','her','hers','daughter'], stimuli_neg=['male','man','boy','brother','he','him','his','son']):
    bge_10k_df = pd.read_csv(path.join("llm_gender_association_collector", "BGE_10k.csv"), na_values=None, keep_default_na=False)
    cohere_10k_df = pd.read_csv(path.join("llm_gender_association_collector", "cohere_10k.csv"), na_values=None, keep_default_na=False)
    openai_10k_df = pd.read_csv(path.join("llm_gender_association_collector", "e5_10k.csv"), na_values=None, keep_default_na=False)
    e5_10k_df = pd.read_csv(path.join("llm_gender_association_collector", "openai_10k.csv"), na_values=None, keep_default_na=False)

    frequency_ceilings = [100, 1000, 10000]

    count_list = []

    for ceiling in frequency_ceilings:
        head_bge_df = bge_10k_df.head(ceiling)
        head_cohere_df = cohere_10k_df.head(ceiling)
        head_openai_df = openai_10k_df.head(ceiling)
        head_e5_df = e5_10k_df.head(ceiling)

        calc_es_percentage(count_list, ceiling, head_bge_df)
        calc_es_percentage(count_list, ceiling, head_cohere_df)
        calc_es_percentage(count_list, ceiling, head_openai_df)
        calc_es_percentage(count_list, ceiling, head_e5_df)

    data = {
        'N': ['10^2', '10^2', '10^2', '10^2', '10^2', '10^2', '10^2', '10^2',
            '10^3', '10^3', '10^3', '10^3', '10^3', '10^3', '10^3', '10^3',
            '10^4', '10^4', '10^4', '10^4', '10^4', '10^4', '10^4', '10^4'],
            #'10^5', '10^5', '10^5', '10^5', '10^5', '10^5', '10^5', '10^5'],
        'Association': [f'BGE {pos_class}', f'BGE {neg_class}', f'Cohere {pos_class}', f'Cohere {neg_class}', f'OpenAI {pos_class}', f'OpenAI {neg_class}', f'e5 {pos_class}', f'e5 {neg_class}',
                        f'BGE {pos_class}', f'BGE {neg_class}', f'Cohere {pos_class}', f'Cohere {neg_class}', f'OpenAI {pos_class}', f'OpenAI {neg_class}', f'e5 {pos_class}', f'e5 {neg_class}',
                        f'BGE {pos_class}', f'BGE {neg_class}', f'Cohere {pos_class}', f'Cohere {neg_class}', f'OpenAI {pos_class}', f'OpenAI {neg_class}', f'e5 {pos_class}', f'e5 {neg_class}'],
                        #f'BGE {pos_class}', f'BGE {neg_class}', f'Cohere {pos_class}', f'Cohere {neg_class}', f'BGE {pos_class}', f'BGE {neg_class}', f'Cohere {pos_class}', f'Cohere {neg_class}'],
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

#all_association_bar_chart()

association_bar_chart_indiv("BGE_10k.csv", "BGE")
association_bar_chart_indiv("cohere_10k.csv", "Cohere")
association_bar_chart_indiv("e5_10k.csv", "e5")
association_bar_chart_indiv("openai_10k.csv", "OpenAI")