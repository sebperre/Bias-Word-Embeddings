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

pd.read_csv(path.join("embeddings", "glove.850B.300d.txt"))