from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
from sklearn import metrics

DATASETS = ['mtat', 'msd', 'jamendo']
MODELS = ['musicnn', 'fcn', 'se']

pd.options.display.precision = 2


def get_results(directory):
    directory = Path(directory)
    num_tracks = {dataset: pd.read_csv(directory / 'analysis' / f'{dataset}.csv', index_col=0) for dataset in DATASETS}
    tags = {dataset: item.index for dataset, item in num_tracks.items()}
    common_tags = sorted(set.intersection(*[set(_tags) for _tags in tags.values()]))
    print(common_tags)

    # ROC_AUC
    performance_results = {}
    for model in MODELS:
        performance_results[model] = {}
        roc_aucs_all = pd.DataFrame(index=DATASETS, columns=DATASETS, dtype='float')
        pr_aucs_all = pd.DataFrame(index=DATASETS, columns=DATASETS, dtype='float')
        for test_dataset in DATASETS:
            performance_results[model][test_dataset] = {}
            for train_dataset in DATASETS:
                path = directory / 'results' / f'test-{test_dataset}-train-{train_dataset}-{model}'
                predictions = pd.read_csv(path / 'est.csv', index_col=0, names=tags[train_dataset])
                groundtruth = pd.read_csv(path / 'gt.csv', index_col=0, names=tags[test_dataset])

                roc_aucs = metrics.roc_auc_score(groundtruth[common_tags], predictions[common_tags], average=None)
                pr_aucs = metrics.average_precision_score(groundtruth[common_tags], predictions[common_tags], average=None)

                print(roc_aucs)
                print(pr_aucs)
                input('break')

                performance_results[model][test_dataset][train_dataset] = \
                    pd.DataFrame()
                roc_aucs_all.loc[train_dataset, test_dataset] = results_run.loc[tags_common, 'AUC'].mean()
        roc_aucs_all.to_csv(directory / 'analysis' / f'{model}.csv', float_format='%.2f')

    # Compare different models on same pair of datasets
    model = 'musicnn'
    for test_dataset in DATASETS:
        for train_dataset in DATASETS:
            if not test_dataset == train_dataset:
                common_tags = num_tracks[test_dataset].index & num_tracks[train_dataset].index
                df = pd.DataFrame(index=common_tags)
                df.loc[:, 'ROC-AUC_cross'] = performance_results[model][test_dataset][train_dataset].loc[common_tags, 'AUC']
                df.loc[:, f'ROC-AUC_{test_dataset}'] = performance_results[model][test_dataset][test_dataset].loc[common_tags, 'AUC']
                df.loc[:, f'ROC-AUC_{train_dataset}'] = performance_results[model][train_dataset][train_dataset].loc[common_tags, 'AUC']

                df.loc[:, f'train_{test_dataset}'] = num_tracks[test_dataset]['train']
                df.loc[:, f'test_{test_dataset}'] = num_tracks[test_dataset]['test']

                df.loc[:, f'train_{train_dataset}'] = num_tracks[train_dataset]['train']
                df.loc[:, f'test_{train_dataset}'] = num_tracks[train_dataset]['test']

                output_directory = directory / 'analysis' / model
                output_directory.mkdir(exist_ok=True)
                df.to_csv(output_directory / f'test-{test_dataset}-train-{train_dataset}.csv', float_format='%.2f')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()

    get_results(args.directory)
