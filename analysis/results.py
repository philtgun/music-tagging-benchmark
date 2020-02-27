from pathlib import Path
from argparse import ArgumentParser
import itertools

import pandas as pd
import numpy as np

DATASETS = ['mtat', 'msd', 'jamendo']
MODELS = ['musicnn', 'fcn', 'se']

pd.options.display.precision = 2


def get_results(directory):
    directory = Path(directory)
    num_tracks = {dataset: pd.read_csv(directory / 'analysis' / f'{dataset}.csv', index_col=0) for dataset in DATASETS}
    tags = {dataset: item.index for dataset, item in num_tracks.items()}

    # ROC_AUC
    performance_results = {}
    for model in MODELS:
        performance_results[model] = {}
        roc_aucs = pd.DataFrame(index=DATASETS, columns=DATASETS, dtype='float')
        for test_dataset in DATASETS:
            performance_results[model][test_dataset] = {}
            for train_dataset in DATASETS:
                results_run = pd.read_csv(
                    directory / 'results' / f'test-{test_dataset}-train-{train_dataset}-{model}' / 'results_report.csv',
                    index_col=0)
                performance_results[model][test_dataset][train_dataset] = results_run
                if test_dataset == train_dataset:
                    tags_common = set.union(*[set(num_tracks[dataset].index & num_tracks[test_dataset].index)
                                              for dataset in DATASETS if not dataset == test_dataset])
                else:
                    tags_common = set(num_tracks[test_dataset].index & num_tracks[train_dataset].index)
                roc_aucs.loc[train_dataset, test_dataset] = results_run.loc[tags_common, 'AUC'].mean()
        roc_aucs.to_csv(directory / 'analysis' / f'{model}.csv', float_format='%.2f')

    # Compare different models on same pair of datasets
    model = 'musicnn'
    for test_dataset in DATASETS:
        for train_dataset in DATASETS:
            if not test_dataset == train_dataset:
                common_tags = num_tracks[test_dataset].index & num_tracks[train_dataset].index
                df = pd.DataFrame(index=common_tags)
                df.loc[:, 'cross_ROC-AUC'] = performance_results[model][test_dataset][train_dataset].loc[common_tags, 'AUC']
                df.loc[:, f'train_{train_dataset}'] = num_tracks[train_dataset]['train']
                df.loc[:, f'test_{test_dataset}'] = num_tracks[test_dataset]['test']

                df.loc[:, 'self_ROC-AUC'] = performance_results[model][test_dataset][test_dataset].loc[common_tags, 'AUC']
                df.loc[:, f'train_{test_dataset}'] = num_tracks[test_dataset]['train']

                output_directory = directory / 'analysis' / model
                output_directory.mkdir(exist_ok=True)
                df.to_csv(output_directory / f'test-{test_dataset}-train-{train_dataset}.csv', float_format='%.2f')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()

    get_results(args.directory)
