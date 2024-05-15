#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import sys
import os
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import normalized_mutual_info_score, rand_score
from .plot_utils import plot_tsne

def eval_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed.
    Allow multiple predicted labels matching one true label.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
        purity, in [0,1]
    """
    assert (y_pred.size == y_true.size), \
        "Incorrect label length in eval_acc! y_pred {}, y_true {}".format(
            y_pred.size, y_true.size)

    # Initialize the confusion matrix as the cost matrix
    D1 = y_pred.max() + 1
    D2 = y_true.max() + 1
    cm = np.zeros((D1, D2), dtype=np.int64)
    for i in range(y_pred.size):
        cm[y_pred[i], y_true[i]] += 1
    
    cv = cm.reshape(-1)  # Reshape to a vector as c in the objective
    # print(D1, D2)

    # Create integer bounds for decision variables
    from scipy import optimize
    bounds = optimize.Bounds(0, 1)
    integrality = np.full_like(cv, True)

    # Create the matrix A and the linear constraint
    A = np.eye(D1).repeat(D2, axis=1)
    contraints = optimize.LinearConstraint(A=A, lb=1, ub=1)

    # Solve the milp problem
    from scipy.optimize import milp
    res = milp(c=cv.max() - cv, 
               constraints=contraints, 
               integrality=integrality, 
               bounds=bounds)

    #print(res.x.reshape(D1, D2))
    # print(w)
    acc = sum(cv * res.x) * 1.0 / y_pred.size

    # compute purity
    purity = np.sum(np.max(cm, axis=0)) / np.sum(cm)
    return acc, purity, cm.T


def eval_nmi(y_true, y_pred):
    """
    Calculate clustering normalized mutial information. Require scikit-learn installed.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        nmi, in [0,1]
    """
    return normalized_mutual_info_score(y_true, y_pred)


def eval_ri(y_true, y_pred):
    """
    Calculate clustering random index. Require scikit-learn installed.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        ri, in [0,1]
    """
    return rand_score(y_true, y_pred)


def tsne_simil(x, metric='euclidean', sigma=1.0):
    dist_matrix = pairwise_distances(x, metric=metric)
    cur_sim = np.divide(- dist_matrix, 2 * sigma ** 2)
    # print(np.sum(cur_sim, axis=1, keepdims=True))

    # mask-out self-contrast cases
    # the diagonal elements of exp_logits should be zero
    logits_mask = np.ones((x.shape[0], x.shape[0]))
    np.fill_diagonal(logits_mask, 0)
    # print(logits_mask)
    exp_logits = np.exp(cur_sim) * logits_mask
    # print(exp_logits.shape)
    # print(np.sum(exp_logits, axis=1, keepdims=True))

    p = np.divide(exp_logits, np.sum(exp_logits, axis=1, keepdims=True) + 1e-10)
    p = p + p.T
    p /= 2 * x.shape[0]
    return p


def cluster_eval(test_embeddings, test_labels, opt, mem, cur_step, epoch, logger):
    """Cluster and plot in evaluations"""
    num_classes = int(np.unique(test_labels).size * opt.k_scale)

    # perform k-means clustering
    st = time.time()
    test_pred_labels = KMeans(n_clusters=num_classes, init='k-means++', n_init=10,
                              max_iter=300, verbose=0).fit_predict(test_embeddings)
    kmeans_time = time.time() - st
    kmeans_acc, kmeans_purity = eval_acc(test_labels, test_pred_labels)
    logger.log_value('kmeans acc', kmeans_acc, cur_step)
    logger.log_value('kmeans purity', kmeans_purity, cur_step)
    print('Val: [{0}][{1}]\t kmeans: acc {acc} purity {purity} (time {time})'.format(
        epoch, cur_step, time=kmeans_time, acc=kmeans_acc, purity=kmeans_purity))
    sys.stdout.flush()
    if opt.plot:
        # plot t-SNE for test embeddings
        plot_tsne(test_embeddings, test_pred_labels, test_labels,
                  title='{} kmeans {}'.format(opt.method, kmeans_acc),
                  fig_name=os.path.join(opt.save_folder, 'kmeans_{}_{}.png'.format(epoch, cur_step)))

    # perform agglomerative clustering
    for metric in ['cosine']:
        st = time.time()
        test_pred_labels = AgglomerativeClustering(
            n_clusters=10, affinity=metric, linkage='average').fit_predict(test_embeddings)
        exec_time = time.time() - st
        agg_acc, agg_purity = eval_acc(test_labels, test_pred_labels)
        logger.log_value('agg {metric} {linkage} acc'.format(metric=metric, linkage='average'),
                         agg_acc, cur_step)
        logger.log_value('agg {metric} {linkage} purity'.format(metric=metric, linkage='average'),
                         agg_purity, cur_step)
        print('Val: [{0}][{1}]\t agg {metric} {linkage}: acc {acc} purity {purity} (time {time})'.format(
            epoch, cur_step, metric=metric, linkage='average', time=exec_time, acc=agg_acc, purity=agg_purity))
        sys.stdout.flush()

        if opt.plot:
            # plot t-SNE for test embeddings
            plot_tsne(test_embeddings, test_pred_labels, test_labels,
                      title='{method} agg {metric} {linkage} {acc}'.format(method=opt.method, metric=metric,
                                                                           linkage='average', acc=agg_acc),
                      fig_name=os.path.join(opt.save_folder,
                                            'agg_{}_{}_{}_{}.png'.format(metric, 'average', epoch, cur_step)))

    # perform spectral clustering
    for metric in ['cosine']:
        st = time.time()
        similarity_matrix = tsne_simil(test_embeddings, metric=metric)
        test_pred_labels = SpectralClustering(n_clusters=num_classes, affinity='precomputed', n_init=10,
                                              assign_labels='discretize').fit_predict(similarity_matrix)
        spectral_time = time.time() - st
        spectral_acc, spectral_purity = eval_acc(test_labels, test_pred_labels)
        logger.log_value('spectral {metric} acc'.format(metric=metric), spectral_acc, cur_step)
        logger.log_value('spectral {metric} purity'.format(metric=metric), spectral_purity, cur_step)
        print('Val: [{0}][{1}]\t spectral {metric}: acc {acc} purity {purity} (time {time})'.format(
            epoch, cur_step, metric=metric, time=spectral_time, acc=spectral_acc, purity=spectral_purity))
        sys.stdout.flush()
        if opt.plot:
            # plot t-SNE for test embeddings
            plot_tsne(test_embeddings, test_pred_labels, test_labels,
                      title='{} spectral {} {}'.format(opt.method, metric, spectral_acc),
                      fig_name=os.path.join(opt.save_folder,
                                            'spectral_{}_{}_{}.png'.format(metric, epoch, cur_step)))

    with open(os.path.join(opt.save_folder, 'result.txt'), 'a+') as f:
        f.write('{epoch},{step},kmeans,{kmeans_acc},agg,{agg_acc},spectral,{spectral_acc},'.format(
            epoch=epoch, step=cur_step, kmeans_acc=kmeans_acc, agg_acc=agg_acc, spectral_acc=spectral_acc
        ))


def knn_eval(test_embeddings, test_labels, knn_train_embeddings, knn_train_labels,
             opt, mem, cur_step, epoch, logger):
    """KNN classification and plot in evaluations"""
    # perform kNN classification
    from sklearn.neighbors import KNeighborsClassifier
    st = time.time()
    neigh = KNeighborsClassifier(n_neighbors=50)
    pred_labels = neigh.fit(knn_train_embeddings, knn_train_labels).predict(test_embeddings)
    knn_time = time.time() - st
    knn_acc = np.sum(pred_labels == test_labels) / pred_labels.size
    logger.log_value('knn acc', knn_acc, cur_step)
    print('Val: [{0}][{1}]\t knn: acc {acc} (time {time})'.format(
        epoch, cur_step, time=knn_time, acc=knn_acc))
    sys.stdout.flush()
    if opt.plot:
        # plot t-SNE for test embeddings
        plot_tsne(test_embeddings, pred_labels, test_labels,
                  title='{} knn {}'.format(opt.method, knn_acc),
                  fig_name=os.path.join(opt.save_folder, 'knn_{}_{}.png'.format(epoch, cur_step)))

    with open(os.path.join(opt.save_folder, 'result.txt'), 'a+') as f:
        f.write('knn,{knn_acc},'.format(knn_acc=knn_acc))


def knn_task_eval(test_embeddings, test_labels, knn_train_embeddings, knn_train_labels,
                  opt, mem, cur_step, epoch, logger, task_list):
    """KNN classification and plot in evaluations"""
    from sklearn.neighbors import KNeighborsClassifier
    st = time.time()
    knn_task_acc = []
    for task in task_list:
        # perform kNN classification
        knn_train_ind = np.isin(knn_train_labels, task)
        test_ind = np.isin(test_labels, task)
        neigh = KNeighborsClassifier(n_neighbors=50)
        pred_labels = neigh.fit(knn_train_embeddings[knn_train_ind],
                                knn_train_labels[knn_train_ind]).predict(test_embeddings[test_ind])
        knn_acc = np.sum(pred_labels == test_labels[test_ind]) / pred_labels.size
        knn_task_acc.append(knn_acc)
    knn_time = time.time() - st
    knn_task_acc = np.mean(knn_task_acc)
    logger.log_value('knn task acc', knn_task_acc, cur_step)
    print('Val: [{0}][{1}]\t knn task: acc {acc} (time {time})'.format(
        epoch, cur_step, time=knn_time, acc=knn_task_acc))
    sys.stdout.flush()

    with open(os.path.join(opt.save_folder, 'result.txt'), 'a+') as f:
        f.write('knn_task,{knn_acc},\n'.format(knn_acc=knn_task_acc))


def eval_forget(acc_mat):
    """
    Evaluate the forgetting measure based on accuracy matrix
    Args:
        acc_mat: numpy array with shape (phase#, class#)
                 acc_mat[i, j] is the accuracy on class j after phase i
    Return:
        a scalar forgetting measure
    """
    forget_pc = acc_mat - acc_mat[-1, :].reshape((1, -1)) # (phase#, class#)
    forget_pc = np.maximum(forget_pc, 0) # Make sure forgetting is positive
    forget_pc = np.max(forget_pc, axis=0) # (class#)
    return np.mean(forget_pc)


def eval_forward_transfer(acc_mat):
    """
    Evaluate the forward transfer based on accuracy matrix
    Args:
        acc_mat: numpy array with shape (phase#, class#)
                 acc_mat[i, j] is the accuracy on class j after phase i
    Return:
        a scalar forward transfer measure
    """
    transfer_pc = np.diagonal(acc_mat, offset=1) - 0.1 # set 10% as acc for random network
    return np.mean(transfer_pc)


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state