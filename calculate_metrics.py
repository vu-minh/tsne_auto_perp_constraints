# calculate quality metric scores

import os
from joblib import Parallel, delayed, load

from metrics import DRMetric
from dataset_utils import load_dataset
import db_utils

import multiprocessing
n_cpus = multiprocessing.cpu_count()
n_cpus_using = int(0.75 * n_cpus)

input_folder = './input'
output_folder = './output'
key_name_in_DB = 'metrics'


def _calculate_metrics(db_name, X_original, tsne_file, metric_names):
    """ Util function that calls `DRMetric` to do the calculation
    """
    print('Processing: ', tsne_file)

    # extract fields from pre-calculated tsne object
    tsne_obj = load(tsne_file)
    X_2d = tsne_obj.embedding_
    loss = tsne_obj.kl_divergence_
    n_iter = tsne_obj.n_iter_
    perp = tsne_obj.get_params()['perplexity']

    # prepare a record to store result
    record = {
        'perp': perp,
        'loss': loss,
        'n_iter': n_iter
    }

    # create a `DRMetric` object that calculates all metric score
    drMetric = DRMetric(X_original, X_2d)

    # calculate each given metric and save the score to `record`
    for metric_name in metric_names:
        metric_method = getattr(drMetric, metric_name)
        record[metric_name] = metric_method()

    # save the result in `record` to database
    db_utils.append_to_db(db_name, key_name_in_DB, records=[record])


def calculate_metric(dataset_name):
    """ Util function that manages to calculate metric scores
        for each given dataset (in parallel)
    """
    db_name = 'DB_{}'.format(dataset_name)

    # prepare a list of metric names:
    metric_names = [
        'auc_rnx',
        'pearsonr',
        'mds_isotonic',
        'cca_stress',
        'sammon_nlm'
    ]

    # load original dataset
    X_original, _, _ = load_dataset(dataset_name)

    # prepare a list of pre-calculated tsne object files
    to_process_files = []
    embedding_dir = '{}/{}'.format(output_folder, dataset_name)
    for file in os.listdir(embedding_dir):
        if file.endswith('.z'):
            in_name = os.path.join(embedding_dir, file)
            to_process_files.append(in_name)
    print('{} files to process'.format(len(to_process_files)))

    # setup to run calculation in parallel
    Parallel(n_jobs=n_cpus_using)(
        delayed(_calculate_metrics)(
            db_name, X_original, tsne_file, metric_names
        ) for tsne_file in to_process_files
    )


if __name__ == '__main__':
    # if run code in sequential model, disable parallel flag
    db_utils.IS_PARALLEL = True

    datasets = ['COUNTRY2014', 'BREAST-CANCER95', 'MPI', 'DIABESTS']
    for dataset_name in datasets:
        calculate_metric(dataset_name)

    # dataset_name = 'MNIST-SMALL'
    # db_name = 'DB_{}'.format(dataset_name)
    # db_utils.show_db(db_name, key='metrics')

    # # clean all data in key
    # db_utils.reset_key(db_name, key='metrics')
    # db_utils.show_db(db_name)
