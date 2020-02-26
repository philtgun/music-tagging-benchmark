from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import numpy as np

DATASETS = ['msd', 'mtat', 'mtg-jamendo']


def get_results(directory):
    num_tracks = {dataset: pd.read_csv(directory / f'{dataset}.csv') for dataset in DATASETS}
    tags = {dataset: set(item.index) for dataset, item in num_tracks.items()}

    tags_msd_mtat = sorted(tags['mtat'] & tags['msd'])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()

    get_results(args.directory)

