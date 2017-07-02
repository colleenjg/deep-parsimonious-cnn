""" run_tsne_and_plot.py

This script runs t-SNE (https://github.com/lvdmaaten/bhtsne).
Choose directory to save in (2x)

Usage:
  run_tsne_and_plot.py
"""

# !/usr/bin/env python2
# -*- coding: utf-8 -*-
import time
import sys

import numpy as np
import cPickle as pickle
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# replace by path to bhtsne
sys.path.append('../../../t-SNE/bhtsne')
import bhtsne


def load_data(filename):
    df = pd.read_csv(filename, header=None)
    df.columns = ['label'] + [str(x) for x in range(1, len(df.columns))]

    # CIFAR10 labels
    label_name_mapping = {0: 'airplane',
                          1: 'automobile',
                          2: 'bird',
                          3: 'cat',
                          4: 'deer',
                          5: 'dog',
                          6: 'frog',
                          7: 'horse',
                          8: 'ship',
                          9: 'truck'}
    df.replace({'label': label_name_mapping}, inplace=True)

    return df


def load_mnist_data(filename):
    """Load the dataset as a pandas DataFrame.

    Download from here: http://deeplearning.net/tutorial/gettingstarted.html
    """
    with open(filename, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    df = pd.DataFrame(np.concatenate((train_set[0], valid_set[0], test_set[0]),
                                     axis=0),
                      columns=[str(x) for x in range(train_set[0].shape[1])])
    df['label'] = np.concatenate((train_set[1], valid_set[1], test_set[1]))
    df['group'] = np.concatenate((np.zeros_like(train_set[1]) + 0,
                                  np.zeros_like(valid_set[1]) + 1,
                                  np.zeros_like(test_set[1]) + 2))

    return df


def scatterplot(df, x, y, c, xlabel, ylabel, title, filename):
    """Make a scatterplot from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with data to plot
        x (str): column name for x axis
        y (str): column name for y axis
        xlabel (str): label of x axis
        ylabel (str): label of y axis
        title (str): title
    """
    fig = sns.lmplot(x, y, data=df, fit_reg=False, hue=c,
                     scatter_kws={'marker': 'D', 's': 5, 'alpha': .5},
                     legend=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper right', frameon=True, markerscale=2,
               fontsize='x-small', labelspacing=.3)
    # # alternative to anchor legend:
    # bbox_to_anchor =(1.05, 1)

    filename = filename + '_' + title

    # choose directory to save in
    fig.savefig('../../t-SNE_results/plots/new/' + filename, dpi=1000)


if __name__ == '__main__':

    # select file to load
    filename = 'mnist.pkl'

    # # directory for activations
    # df = load_data('../../layer_activations/' + filename)

    # # directory for raw or shuffled raw CIFAR10 images
    # df = load_mnist_data('../../cifar_raw_data_modified/' + filename)

    # directory for raw MNIST data
    df = load_mnist_data('../../../Datasets/' + filename)

    # apply PCA and put first components in dataframe
    n_components = 50
    pca = PCA(n_components=n_components)

    dataset_pcs = pca.fit_transform(df.drop(['label'], axis=1))
    pcs_df = pd.DataFrame(dataset_pcs, columns=[str(x) for x in range(50)])
    pcs_df['label'] = df['label']

    pcs_df_alp = pcs_df.sort_values('label')

    if filename.endswith(('.txt', '.pkl')):
        filename = filename[:-4]

    # plot first 2 PCs
    scatterplot(pcs_df_alp, '0', '1', 'label', 'PC1', 'PC2', 'First two PCs',
                filename)

    # run t-SNE on PCA data
    print('Running t-SNE!')
    t = time.time()
    embedding_array = bhtsne.run_bh_tsne(dataset_pcs,
                                         initial_dims=dataset_pcs.shape[1])
    elapsed = time.time() - t
    print('Done. Elapsed time: ', elapsed)

    tsne_df = pd.DataFrame(embedding_array, columns=['0', '1'])
    tsne_df['label'] = pcs_df['label']

    tsne_df_alp = tsne_df.sort_values('label')

    my_df = pd.DataFrame(tsne_df_alp)

    # choose directory to save in
    my_df.to_csv('../../t-SNE_results/values/new/' + filename + '_tsne_df.txt',
                 index=False, header=True)
    print('t-SNE df file saved')

    # plot t-SNE
    scatterplot(tsne_df_alp, '0', '1', 'label', '1', '2', 't-SNE', filename)

    # prevent closing
    plt.waitforbuttonpress()
