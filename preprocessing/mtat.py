import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm


def _save_ids_from_npy(directory: Path, input_filename, output_filename):
    data = np.load(directory / input_filename)
    ids = [row.split('\t')[0] for row in data]
    df = pd.DataFrame(ids)
    df.to_csv(directory / output_filename, index=False, header=False)


def process(directory):
    directory = Path(directory)
    tags = np.load(directory / 'tags.npy')
    data = np.load(directory / 'binary.npy')

    tags_df = pd.DataFrame(data, columns=tags)
    tags_df.to_csv(directory / 'tags.csv')

    _save_ids_from_npy(directory, 'train.npy', 'train.txt')
    _save_ids_from_npy(directory, 'valid.npy', 'validation.txt')
    _save_ids_from_npy(directory, 'test.npy', 'test.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()

    process(args.directory)
