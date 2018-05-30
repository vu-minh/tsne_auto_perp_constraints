# calculate constraint-preserving scores

import os
import random
import numpy as np

from joblib import Parallel, delayed, load
from scipy.spatial.distance import pdist, squareform

from dataset_utils import load_dataset
import db_utils

import multiprocessing
n_cpus = multiprocessing.cpu_count()
n_cpus_using = int(0.75 * n_cpus)

MACHINE_EPSILON = np.finfo(np.double).eps

input_folder = './input'
output_folder = './output'
key_name_in_DB = 'constraints'

# the different number of pairs for each constraint type
num_constraints = [1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100, 200, 500]
# for each number of constraint, repeat the generation process to make different pairs
n_repeats = 10


def _generate_constraints(labels, n_take, reproduce_seed):
    """ Generate the pairwise constraints based on labels
    Make sure to obtain different pairs each time this function is called
    """
    # print('Generate {} links with seed {}'.format(n_take, reproduce_seed))
    random.seed(reproduce_seed)
    n_samples = len(labels)
    must_links = []
    cannot_links = []

    # make pairs until reaching enough quantity
    while len(must_links) < n_take or len(cannot_links) < n_take:
        i1 = random.randint(0, n_samples - 1)
        i2 = random.randint(0, n_samples - 1)
        if i1 == i2:
            continue

        if labels[i1] == labels[i2]:
            must_links.append([i1, i2])
        else:
            cannot_links.append([i1, i2])

    random.shuffle(must_links)
    random.shuffle(cannot_links)
    return must_links[:n_take], cannot_links[:n_take]


def _compute_Q(X2d):
    """ Matrix Q in t-sne, used to calculate the prob. that a point `j`
    being neighbor of a point `i` (the value of Q[i,j])
    """
    degrees_of_freedom = 1
    X2d = X2d.reshape(-1, 2)

    dist = pdist(X2d, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
    return squareform(Q)


def _constraint_score(Q, mls, cls):
    """ Constraint preserving score, in fact it is the log likelihood that
    two points in a mustlink are close:
    $$ S_{\mathcal{M}}
        = \frac{1}{|\mathcal{M}|} \log \prod_{\mathcal{M}} q_{ij}
        = \frac{1}{|\mathcal{M}|} \sum_{\mathcal{M}} \log q_{ij}
    $$,
    or two points in a cannot-link are far apart:
    $$ S_{\mathcal{C}}
        = \frac{-1}{|\mathcal{C}|} \log \prod_{\mathcal{C}} q_{ij}
        = \frac{-1}{|\mathcal{C}|} \sum_{\mathcal{C}} \log q_{ij}
    $$
    """
    mls = np.array(mls)
    cls = np.array(cls)
    s_ml = np.sum(np.log(Q[mls[:, 0], mls[:, 1]])) / len(mls)
    s_cl = - np.sum(np.log(Q[cls[:, 0], cls[:, 1]])) / len(cls)
    return s_ml, s_cl


def _calculate_constraint_score(db_name, labels, tsne_file):
    print('Processing: ', tsne_file)

    # extract fields from pre-calculated tsne object
    tsne_obj = load(tsne_file)
    X_2d = tsne_obj.embedding_
    perp = tsne_obj.get_params()['perplexity']

    # prepare matrix Q in low dim.
    Q = _compute_Q(X_2d)

    # repeat the constraint generation process
    for n_take in num_constraints:
        records = []
        for _ in range(n_repeats):
            random.seed(None)
            reproduce_seed = random.randint(0, 1e10)
            mls, cls = _generate_constraints(labels, n_take, reproduce_seed)
            s_ml, s_cl = _constraint_score(Q, mls, cls)

            a_record = {
                'perp': perp,
                'n_constraints': n_take,
                's_ml': s_ml,
                's_cl': s_cl,
                's_all': s_ml + s_cl,
                'reproduce_seed': reproduce_seed
            }
            records.append(a_record)
        # save to db the scores according to `n_take`
        db_utils.append_to_db(db_name, key_name_in_DB, records=records)


def _calculate_manual_constraint(db_name, labels, tsne_file):
    print('Processing: ', tsne_file)

    # extract fields from pre-calculated tsne object
    tsne_obj = load(tsne_file)
    X_2d = tsne_obj.embedding_
    perp = tsne_obj.get_params()['perplexity']

    # prepare matrix Q in low dim.
    Q = _compute_Q(X_2d)

    dataset_name = db_name[3:]
    n_constraints = 10
    mls, cls = _load_manual_constraint(dataset_name, n_constraints)
    s_ml, s_cl = _constraint_score(Q, mls, cls)
    a_record = {
        'perp': perp,
        'n_constraints': n_constraints,
        's_ml': s_ml,
        's_cl': s_cl,
        's_all': s_ml + s_cl,
        'reproduce_seed': -1
    }
    db_utils.append_to_db(db_name, records=[a_record],
                          key_name_in_DB='manual_constraints')


def _load_manual_constraint(dataset_name, n_take=10):
    path = './output/manual_constraints/{}.pkl'.format(dataset_name)
    pickle_obj = load(path)
    mls = pickle_obj['mustlinks']
    cls = pickle_obj['cannotlinks']
    return mls[:n_take], cls[:n_take]


def calculate_constraint_score(dataset_name, auto_constraint=True):
    if auto_constraint:
        calculation_function = _calculate_constraint_score
    else:
        calculation_function = _calculate_manual_constraint

    # prepare database name and original labeled data
    db_name = 'DB_{}'.format(dataset_name)
    _, labels, _ = load_dataset(dataset_name)

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
        delayed(calculation_function)(
            db_name, labels, tsne_file
        ) for tsne_file in to_process_files
    )


if __name__ == '__main__':
    db_utils.IS_PARALLEL = True

    # datasets = ['BREAST-CANCER95', 'MPI', 'DIABETES']
    # for dataset_name in datasets:
    #     print('Calculate Constraint scores for ', dataset_name)
    #     calculate_constraint_score(dataset_name)

    dataset_name = 'COUNTRY-2014'
    calculate_constraint_score(dataset_name='COUNTRY-2014',
                               auto_constraint=False)

    db_name = 'DB_{}'.format(dataset_name)
    db_utils.show_db(db_name, key='manual_constraints')
