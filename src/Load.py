import sys
import os
import pandas as pd

def load():
    input_file = '/Users/agos/Documents/Echo-Chambers/MyProject/Datasets/brexit_stances.tsv'
    df = pd.read_csv(input_file, sep='\t', header=0, usecols=['tweets', 'stance'])
    return df