""" generate_plots.py

Make multipanelled plots with t-SNE results for activations.

Usage:
  generate_plots.py
"""

# !/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_activ_data(filename):
    df = pd.read_csv(filename, header=0)

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


def scatterplot(df, x, y, c, titles):
    """Make a scatterplot from a DataFrame.

    Args:
        df (list of pd.DataFrame): list of DataFrames with data to plot
        x (str): df column name for x axis
        y (str): df column name for y axis
        c: df column name for colours
        xlabel (str): label of x axis
        ylabel (str): label of y axis
        titles (list of str): list of titles
    """
    all_df = pd.concat(df, axis=0, ignore_index=True)
    all_df.rename(columns={'0': ' ', '1': '  ', 'label': '   '}, inplace=True)

    types = ['Baseline', 'Sample-clustering', 'Baseline', 'Sample-clustering']
    sizes = ['Cumbersome', 'Cumbersome', 'Distilled', 'Distilled']

    # add type and size properties to dataframe
    all_df['Type'] = [y1 for i, y1 in enumerate(types)
                      for x1 in range(df[i].shape[0])]
    all_df['Size'] = [y1 for i, y1 in enumerate(sizes)
                      for x1 in range(df[i].shape[0])]

    plot = sns.lmplot(x=' ', y='  ', data=all_df, fit_reg=False, hue='   ',
                      row='Size', col='Type',
                      scatter_kws={'marker': 'D', 's': 4, 'alpha': .5},
                      legend=False)

    # plt.title(title)
    plt.legend(bbox_to_anchor=(1, .63), bbox_transform=plt.gcf().transFigure,
               frameon=True, markerscale=5, fontsize='small', labelspacing=.5)
    plt.tight_layout(rect=[0, 0, 0.90, 1])


if __name__ == '__main__':

    # load dataframes
    plot_base = load_activ_data(filename='../../t-SNE_results/values/Activations_layer_4_CIFAR10_baseline_model_snap_8_tsne_df.txt')
    plot_clust = load_activ_data(filename='../../t-SNE_results/values/Activations_layer_4_CIFAR10_sample_clustering_model_snap_6_tsne_df.txt')
    plot_base_dist = load_activ_data(filename='../../t-SNE_results/values/Activations_layer_4_CIFAR10_distilled_baseline_model_snap_2_tsne_df.txt')
    plot_clust_dist = load_activ_data(filename='../../t-SNE_results/values/Activations_layer_4_CIFAR10_hybrid_sample_model_snap_4_tsne_df.txt')

    tsne_dfs = [plot_base, plot_clust, plot_base_dist, plot_clust_dist]
    tsne_titles = ['Baseline', 'Sample-clustering', 'Distilled-baseline',
                   'Distilled-sample-clustering']

    # plot t-SNE for activations
    scatterplot(tsne_dfs, '0', '1', 'label', tsne_titles)

    # prevent closing
    plt.waitforbuttonpress()
