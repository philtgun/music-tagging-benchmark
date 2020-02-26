import argparse
from pathlib import Path
import pickle

from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

SUBSETS = ['train', 'validation', 'test']


def get_stats(directory):
    directory = Path(directory)
    df_all = None
    for subset in SUBSETS:
        df = pd.concat([pd.read_csv(path, delimiter='\t', index_col='tag') for path in (directory / subset).iterdir()])
        if df_all is None:
            df_all = df
        df_all[subset] = df['tracks']
    df_all = df_all.sort_index()
    df_all = df_all[SUBSETS]
    df_all.index.name = None
    df_all.to_csv(directory / 'stats.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()

    get_stats(args.directory)
