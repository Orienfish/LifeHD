#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import warnings

labels = {
    'cifar10': ['airplane', 'automobile', 'bird','cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck'],
    'cifar100': ['aquatic mammals', 'fish', 'flowers', 'food containers',
                 'fruit and vegetables', 'household electrical device',
                 'household furniture', 'insects', 'large carnivores',
                 'large man-made outdoor things', 
                 'large natural outdoor scenes',
                 'large omnivores and herbivores',
                 'medium-sized mammals', 
                 'non-insect invertebrates',
                 'people', 'reptiles', 'small mammals', 'trees',
                 'vehicles 1', 'vehicles 2'],
    'esc50': ['Dog', 'Rooster', 'Pig', 'Cow', 'Frog',
              'Cat', 'Hen', 'Insects', 'Sheep', 'Crow',
              'Rain', 'Sea waves', 'Crackling fire', 'Crickets',
              'Chirping birds', 'Water drops', 'Wind', 
              'Pouring water', 'Toilet flush', 'Thunderstrom',
              'Crying baby', 'Sneezing', 'Clapping',
              'Breathing', 'Coughing', 'Footsteps',
              'Laughing', 'Brushing teeth', 'Snoring',
              'Drinking/sipping', 'Door knock', 'Mouse click',
              'Keyboard typing', 'Door, wood creaks',
              'Can opening', 'Washing machine', 'Vacuum cleaner',
              'Clock alarm', 'Clock tick', 'Glass breaking',
              'Helicopter', 'Chainsaw', 'Siren', 'Car horn',
              'Engine', 'Train', 'Church bells', 'Airplane',
              'Fireworks', 'Handsaw'],
    'mhealth': ['Standing still', 'Sitting and relaxing',
                'Lying down', 'Walking', 'Climbing stairs',
                'Waist bends forward', 'Frontal elevation of arms',
                'Knees bending', 'Cycling', 'Jogging', 'Running',
                'Jump front & back'],
    'har': ['Walking', 'Walking upstaris', 'Walking downstairs',
            'Sitting', 'Standing', 'Laying'],
    'har_timeseries': ['Walking', 'Walking upstaris', 'Walking downstairs',
                       'Sitting', 'Standing', 'Laying']
}

def plot_confusion_matrix(cm, dataset, save_folder):
    """
    Plot confusion matrix
    Args:
        cm: confusion matrix of size (D_true, D_pred)
            Note, D_true is the number of true classes, 
            D_pred is the number of predicted classes.
        save_folder: the path to the folder to save plots
    """
    print(cm.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # annot=True to annotate cells, ftm='g' to disable scientific notation
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted Labels by LifeHD')
    ax.set_ylabel('True Labels in MHEALTH')
    #ax.xaxis.set_ticklabels(np.arange(cm.shape[0]))
    #ax.yaxis.set_ticklabels(labels[dataset], rotation=0)
    plt.rcParams.update({'font.size': 10})
    plt.savefig(os.path.join(save_folder, 'cm.png'), dpi=300,
                bbox_inches='tight')


def plot_novelty_detection(x_shift, x_detect, save_folder):
    print('class shift at: ', x_shift)
    print('novelty detect at: ', x_detect)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for x in x_shift:
        if x_shift.index(x) == 0:
            plt.axvline(x=x, color='tab:blue', label='class shift')
        else:
            plt.axvline(x=x, color='tab:blue')
    for x in x_detect:
        if x_detect.index(x) == 0:
            plt.axvline(x=x, color='tab:orange', label='novelty detection')
        else:
            plt.axvline(x=x, color='tab:orange')

    plt.rcParams.update({'font.size': 14})
    ax.set_xlabel(r'Training Steps', size=14)
    # ax1.set_ylabel('Avg ACC', size=15)
    ax.axes.get_yaxis().set_visible(False)
    plt.legend(fontsize=14)
    # plt.xticks(x1)
    plt.savefig(os.path.join(save_folder, 'novelty_detect.png'))


def plot_tsne_graph(x, k=3, title='', fig_name=''):
    """
    Plot the TSNE of x, assigned with true labels and pseudo labels respectively.
    Args:
        x: (batch_size, input_dim), raw data to be plotted
        k: Number of neighbors for each sample.
        title: str, title for the plots
        fig_name: str, the file name to save the plot
    """
    # Obtain the k nearest neighbor graph using Euclidean distance
    L = kneighbors_graph(x, k, include_self=True).toarray()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(2, perplexity=50, learning_rate='auto', init='pca')
    x_emb = tsne.fit_transform(x)

    # Create position node and edges
    G = nx.Graph()
    pos = {}
    bsz = x.shape[0]
    for i in range(bsz):
        G.add_node(i)
        pos[i] = (x_emb[i, 0], x_emb[i, 1])
    
    for i in range(bsz):
        for j in range(bsz):
            if L[i, j]:
                G.add_edge(i, j)

    # print(pos)
    plt.figure()
    nx.draw_networkx(G, pos=pos)
    plt.title(title)
    if fig_name != '':
        plt.savefig(fig_name, dpi=300)


def plot_pca(x, y_pred, y_true=None, title='', fig_name=''):
    """
    Plot the TSNE of x, assigned with true labels and pseudo labels respectively.
    Args:
        x: (batch_size, input_dim), raw data to be plotted
        y_pred: (batch_size), optional, pseudo labels for x
        y_true: (batch_size), ground-truth labels for x
        title: str, title for the plots
        fig_name: str, the file name to save the plot
    """
    pca = PCA(n_components=2)
    x_emb = pca.fit_transform(x)

    if y_true is not None: # Two subplots
        fig = plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(121)
        sns.scatterplot(x=x_emb[:, 0], y=x_emb[:, 1], hue=y_pred,
                        palette=sns.color_palette("hls", np.unique(y_pred).size),
                        legend="full", ax=ax1)
        ax1.set_title('Pred labels, {}'.format(title))
        ax2 = plt.subplot(122)
        sns.scatterplot(x=x_emb[:, 0], y=x_emb[:, 1], hue=y_true,
                        palette=sns.color_palette("hls", np.unique(y_true).size),
                        legend="full", ax=ax2)
        ax2.set_title('True labels, {}'.format(title))
    else: # Only one plot for predicted labels
        fig = plt.figure(figsize=(6, 5))
        sns.scatterplot(x=x_emb[:, 0], y=x_emb[:, 1],
                        hue=y_pred, palette=sns.color_palette("hls", np.unique(y_pred).size),
                        legend="full")
        # plt.title('Pred labels, {}'.format(title))
    plt.rcParams.update({'font.size': 14})

    if fig_name != '':
        plt.savefig(fig_name, bbox_inches='tight')

    plt.close(fig)


def plot_tsne(x, y_pred, y_true=None, title='', fig_name=''):
    """
    Plot the TSNE of x, assigned with true labels and pseudo labels respectively.
    Args:
        x: (batch_size, input_dim), raw data to be plotted
        y_pred: (batch_size), optional, pseudo labels for x
        y_true: (batch_size), ground-truth labels for x
        title: str, title for the plots
        fig_name: str, the file name to save the plot
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(2, perplexity=50, learning_rate='auto', init='pca')
    x_emb = tsne.fit_transform(x)

    if y_true is not None: # Two subplots
        fig = plt.figure(figsize=(12, 5))
        ax1 = plt.subplot(121)
        sns.scatterplot(x=x_emb[:, 0], y=x_emb[:, 1], hue=y_pred,
                        palette=sns.color_palette("hls", np.unique(y_pred).size),
                        legend="full", ax=ax1)
        ax1.set_title('Pred labels, {}'.format(title))
        ax2 = plt.subplot(122)
        sns.scatterplot(x=x_emb[:, 0], y=x_emb[:, 1], hue=y_true,
                        palette=sns.color_palette("hls", np.unique(y_true).size),
                        legend="full", ax=ax2)
        ax2.set_title('True labels, {}'.format(title))
    else: # Only one plot for predicted labels
        fig = plt.figure(figsize=(6, 5))
        sns.scatterplot(x=x_emb[:, 0], y=x_emb[:, 1],
                        hue=y_pred, palette=sns.color_palette("hls", np.unique(y_pred).size),
                        legend="full")
        # plt.title('Pred labels, {}'.format(title))
    plt.rcParams.update({'font.size': 14})

    if fig_name != '':
        plt.savefig(fig_name, bbox_inches='tight')

    plt.close(fig)


def plot_select_tsne(all_embeddings, all_true_labels, select_indexes,
                title='', fig_name=''):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(2, perplexity=50, learning_rate='auto', init='pca', )
    x_emb = tsne.fit_transform(all_embeddings)

    fig = plt.figure(figsize=(6, 5))
    sns.scatterplot(x=x_emb[:, 0], y=x_emb[:, 1],
                    hue=all_true_labels,
                    palette=sns.color_palette("hls",
                                              np.max(all_true_labels)+1),
                    legend="full", alpha=0.2)
    sns.scatterplot(x=x_emb[select_indexes, 0], y=x_emb[select_indexes, 1],
                    hue=all_true_labels[select_indexes],
                    palette=sns.color_palette("hls",
                                              np.max(all_true_labels[select_indexes])+1),
                    legend=False)
    plt.title('Embeddings, {}'.format(title))
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close(fig)

