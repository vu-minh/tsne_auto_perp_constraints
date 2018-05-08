# calculate quality metric scores

import joblib
import os


from metrics import DRMetric
from dataset_utils import load_dataset
import db_utils


input_folder = './input'
output_folder = './output'


def _calculate_metrics(X_original, tsne_obj, metric_names):
    """ Util function that calls `DRMetric` to do the calculation
    """

    # extract fields from pre-calculated tsne object
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

    return record


def calculate_metric(dataset_name):
    """ Util function that manages all steps to calculate metric score
        for each given dataset
    """

    # prepare database and key to store the result
    db_name = 'DB_{}'.format(dataset_name)
    key_name = 'metrics'

    # prepare a list of metric names:
    metric_names = [
        'auc_rnx',
        'pearsonr',
        'mds_isotonic',
        'cca_stress',
        'sammon_nlm'
    ]

    # load original dataset
    X_original, y, target_labels = load_dataset(dataset_name)

    # prepare a loop to load all pre-defined tsne objects
    embedding_dir = '{}/{}'.format(output_folder, dataset_name)
    for file in os.listdir(embedding_dir):
        if file.endswith('.z'):
            in_name = os.path.join(embedding_dir, file)
            print('Processing: ', in_name)

            tsne_obj = joblib.load(in_name)
            record = _calculate_metrics(X_original, tsne_obj, metric_names)

            # save the calculated metric scores to HDFStore
            db_utils.append_to_db(db_name, key_name, records=[record])


if __name__ == '__main__':
    # if run code in sequential model, disable parallel flag
    db_utils.IS_PARALLEL = True

    dataset_name = 'COIL20'
    db_name = 'DB_{}'.format(dataset_name)

    calculate_metric(dataset_name)
    db_utils.show_db(db_name, key='metrics')

    # # clean all data in key
    # db_utils.reset_key(db_name, key='metrics')
    # db_utils.show_db(db_name)
