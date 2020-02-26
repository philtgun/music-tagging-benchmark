import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def get_num_examples(path, data, tags):
    subset = np.loadtxt(path, dtype=str)
    return np.count_nonzero(data.loc[subset, tags], axis=0)


def get_stats(directory, tags=None):
    directory = Path(directory)
    data = pd.read_csv(directory / 'tags.csv', index_col=0)
    data.index = data.index.astype(str)

    if tags is None:
        tags = data.columns
    train_stats = get_num_examples(directory / 'train.txt', data, tags)
    validation_stats = get_num_examples(directory / 'validation.txt', data, tags)
    test_stats = get_num_examples(directory / 'test.txt', data, tags)

    results = pd.DataFrame(np.array([train_stats, validation_stats, test_stats]).transpose(), index=tags,
                           columns=['train', 'validation', 'test'])
    results = results.sort_index()
    results.to_csv(directory / 'stats.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()

    get_stats(args.directory)

