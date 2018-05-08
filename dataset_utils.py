import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from functools import partial
import pickle

input_folder = './input'
output_folder = './output'
fixed_random_seed = 0


def load_dataset(name='MNIST-SMALL'):
    print('Loading dataset: {}'.format(name))
    return {
        'COIL20': load_coil20,
        'MNIST': partial(load_mnist_full, 6000),
        'MNIST-SMALL': load_mnist_mini,
        'WIKI-FR-1K': partial(load_wiki, 'fr'),
        'WIKI-EN-1K': partial(load_wiki, 'en'),
        'WIKI-FR-3K': partial(load_wiki, 'fr', 3000),
        'WIKI-EN-3K': partial(load_wiki, 'en', 3000),
        'COUNTRY1999': partial(load_country, 1999),
        'COUNTRY2013': partial(load_country, 2013),
        'COUNTRY2014': partial(load_country, 2014),
        'COUNTRY2015': partial(load_country, 2015),
        'CARS04': partial(load_pickle, 'cars04'),
        'BREAST-CANCER95': partial(load_pickle, 'breastCancer'),
        'DIABETES': partial(load_pickle, 'diabetes'),
        'MPI': partial(load_pickle, 'MPI_national'),
        'INSURANCE': partial(load_pickle, 'insurance'),
        'FIFA18': partial(load_pickle, 'fifa18', 2000),
        'FR_SALARY': partial(load_pickle, 'FR_net_salary', 2000),
    }[name]()


def load_coil20():
    import scipy.io
    mat = scipy.io.loadmat("{}/COIL20.mat".format(input_folder))
    X, y = mat['X'], mat['Y'][:, 0]
    X, y = shuffle(X, y, n_samples=len(y), random_state=fixed_random_seed)
    labels = list(map(str, y.tolist()))
    return X, y, labels


def load_mnist_mini():
    dataset = datasets.load_digits()
    X, y = dataset.data, dataset.target
    labels = list(map(str, range(len(y))))
    return X, y, labels


def load_mnist_full(n_samples=2000):
    from sklearn.datasets import fetch_mldata
    dataset = fetch_mldata('MNIST original', data_home=input_folder)
    X, y = dataset.data, dataset.target
    X, y = shuffle(X, y, n_samples=n_samples, random_state=fixed_random_seed)
    y = y.astype(int)
    labels = list(map(str, range(len(y))))
    return X, y, labels


def load_pickle(name, limit_size=2000):
    inputName = '{}/{}.pickle'.format(input_folder, name)
    dataset = pickle.load(open(inputName, 'rb'))
    X, labels = dataset['data'], dataset['labels']
    n = min(limit_size, X.shape[0])
    X = X[:n]
    labels = labels[:n]
    if 'y' in dataset:
        y = dataset['y'][:n]
    else:
        y = np.zeros(n)
    # print("Data from pickle: ", X.shape, y.shape, len(labels))
    return X, y, labels


def load_wiki(lang='en', n=1000): return load_pickle(
    name='wiki_{}_n{}_d300'.format(lang, n))


def load_country(year): return load_pickle(
    name='country_indicators_{}'.format(year))
