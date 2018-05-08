# run tsne for different datasets with different perplexity

import time
from joblib import Parallel, delayed, dump
from sklearn.manifold import TSNE

from utils import gen_log_space_float
from dataset_utils import load_dataset

import multiprocessing
n_cpus = multiprocessing.cpu_count()
n_cpus_using = int(0.75 * n_cpus)

out_dir = './output'

def _run_tsne(X, perp, i):
    print('[i={}] perp={}'.format(i, perp))
    tsne = TSNE(perplexity=perp, random_state=0, verbose=0)
    tsne.fit_transform(X)
    out_name = '{}/{}/{:04d}.z'.format(out_dir, dataset_name, i)
    dump(tsne, out_name)

def run_embedding(dataset_name):
    X, y, _ = load_dataset(dataset_name)
    N = X.shape[0]
    n_perps = min(1000, N//2)
    perps = gen_log_space_float(limit=N, n=n_perps)
    print('Number of perps: ', len(perps))
    
    Parallel(n_jobs=n_cpus_using)(
        delayed(_run_tsne)(X, perp, i) for i, perp in enumerate(perps)
    )


if __name__ == '__main__':
    dataset_name = 'MNIST'
    run_embedding(dataset_name)

