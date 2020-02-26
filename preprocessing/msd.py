import argparse
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

MSD_CUTOFF = 201680


def _load_pickle(path: Path):
    with path.open('rb') as fp:
        return pickle.load(fp, encoding='bytes')


def _save_pickle_list(data, filename):
    data = map(lambda x: x.decode(), data)
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, header=False)


def process(directory):
    directory = Path(directory)
    data_pickle = _load_pickle(directory / 'msd_id_to_tag_vector.cP')

    data = {}
    for key, value in tqdm(data_pickle.items()):
        data[key.decode()] = value.flatten().astype(int)

    tags = np.loadtxt(directory / '50tagList.txt', dtype=str, delimiter='\n')
    tags = np.char.lower(tags)

    tags_df = pd.DataFrame.from_dict(data, orient='index', columns=tags)
    tags_df.to_csv(directory / 'tags.csv')

    train_list = _load_pickle(directory / 'filtered_list_train.cP')
    _save_pickle_list(train_list[:MSD_CUTOFF], directory / 'train.txt')
    _save_pickle_list(train_list[MSD_CUTOFF:], directory / 'validation.txt')

    test_list = _load_pickle(directory / 'filtered_list_test.cP')
    _save_pickle_list(test_list, directory / 'test.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()

    process(args.directory)
