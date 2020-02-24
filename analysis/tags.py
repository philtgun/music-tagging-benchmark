import argparse
from pathlib import Path
import pickle

from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

JAMENDO_TAGS = ['alternative', 'ambient', 'atmospheric', 'chillout', 'classical', 'dance', 'downtempo', 'easylistening',
                'electronic', 'experimental', 'folk', 'funk', 'hiphop', 'house', 'indie', 'instrumentalpop', 'jazz',
                'lounge', 'metal', 'newage', 'orchestral', 'pop', 'popfolk', 'poprock', 'reggae', 'rock', 'soundtrack',
                'techno', 'trance', 'triphop', 'world', 'acousticguitar', 'bass', 'computer', 'drummachine', 'drums',
                'electricguitar', 'electricpiano', 'guitar', 'keyboard', 'piano', 'strings', 'synthesizer', 'violin',
                'voice', 'emotional', 'energetic', 'film', 'happy', 'relaxing']
MTAT_TAGS = ['guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast', 'piano',
             'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'indian', 'opera', 'male', 'singing', 'vocals',
             'no vocals', 'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male vocal', 'no vocal', 'pop', 'soft',
             'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age', 'dance', 'male voice', 'female vocal',
             'beats', 'harp', 'cello', 'no voice', 'weird', 'country', 'metal', 'female voice', 'choral']
MSD_TAGS = ['rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists', 'dance', '00s', 'alternative rock',
            'jazz', 'beautiful', 'metal', 'chillout', 'male vocalists', 'classic rock', 'soul', 'indie rock', 'mellow',
            'electronica', '80s', 'folk', '90s', 'chill', 'instrumental', 'punk', 'oldies', 'blues', 'hard rock',
            'ambient', 'acoustic', 'experimental', 'female vocalist', 'guitar', 'hip-hop', '70s', 'party', 'country',
            'easy listening', 'sexy', 'catchy', 'funk', 'electro', 'heavy metal', 'progressive rock', '60s', 'rnb',
            'indie pop', 'sad', 'house', 'happy']

TAGS = {
    'jamendo': JAMENDO_TAGS,
    'mtat': MTAT_TAGS,
    'msd': MSD_TAGS
}

pd.option_context('display.float_format', '{:0.3f}'.format)


def get_stats_test(directory, train_dataset, test_dataset):
    predictions = pd.read_csv(directory / 'est.csv', header=None, index_col=0)
    groundtruth = pd.read_csv(directory / 'gt.csv', header=None, index_col=0)

    common_tags = list(set(TAGS[train_dataset]) & set(TAGS[test_dataset]))

    examples_per_label = np.count_nonzero(groundtruth, axis=0)
    examples_per_label_df = pd.DataFrame(examples_per_label, index=TAGS[test_dataset], columns=['n_test'])
    examples_per_label_df = examples_per_label_df.loc[common_tags]
    return examples_per_label_df
    # examples_per_label_df.to_csv(directory / 'n_examples_per_label.csv')


def get_num_examples(path, data, tags):
    subset = np.loadtxt(path, dtype=str)
    return np.count_nonzero(data.loc[subset, tags], axis=0)


def get_stats(directory, tags):
    directory = Path(directory)
    data = pd.read_csv(directory / 'tags.csv', index_col=0)

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
    # parser.add_argument('model')
    args = parser.parse_args()

    mtat_jamendo_tags = list(set(TAGS['msd']) & set(TAGS['jamendo']))
    get_stats(args.directory, mtat_jamendo_tags)
    # datasets = ['jamendo', 'msd', 'mtat']
    # for _train_dataset in datasets:
    #     for _test_dataset in datasets:
    #         _directory = Path(args.directory) / f"test-{_test_dataset}-train-{_train_dataset}-{args.model}"
    #         print(f'Analysing directory {_directory}')
    #         stats_test(Path(_directory), _train_dataset, _test_dataset)
