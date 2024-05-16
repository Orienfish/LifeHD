from __future__ import print_function

import os
import copy
import numpy as np
import sys
import argparse
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from tqdm import tqdm
from utils.eval_utils import eval_acc, eval_nmi, eval_ri
from utils.plot_utils import plot_tsne, plot_tsne_graph, plot_novelty_detection

novelty_detect = []
class_shift = []
VAL_CNT = 10
MERGE_CNT = 20


def get_nc_laplacian(class_hvs, batch_idx, opt):
    """
    Obtain the number of clusters by searching for plateaus
    in the sorted eigenvalues

    Args:
        class_hvs: extracted class hypervectors by calling model.extract_class_hv()
        batch_idx: the batch index in the training stream for logging
        opt: arguments

    Returns:
        nc: the number of clusters as the start of the plateau
        L: the k neighbors graph of the input class_hvs
        U: the eigenvectors of L
    """
    G = kneighbors_graph(class_hvs, 3, include_self=True).toarray()
    L = csgraph.laplacian(G)
    # print(L)

    # Compute the eigenvalues and eigenvectors of L
    (S, U) = np.linalg.eig(G)
    S, U = np.real(S), np.real(U)
    ixs = np.argsort(S)  # Sort, ascending
    S, U = S[ixs], U[:, ixs]
    U = U[:, S > 0]
    S = S[S > 0]
    S = S / S.max()

    if batch_idx == opt.warmup_batches or batch_idx % 50 == 0:
        # Plot sorted eigenvalues
        fig = plt.figure()
        plt.plot(np.arange(S.size), S)
        plt.title('Idx: {}'.format(
            batch_idx
        ))
        plt.savefig(os.path.join(opt.save_folder, 'eigenvalue_{}.png'.format(batch_idx)))
        plt.close(fig)

        # Plot the eigenvector corresponding to the second
        # smallest eigenvalue
        fig = plt.figure()
        plt.plot(np.arange(U.shape[0]), np.sort(U[:, 1]))
        plt.title('Idx: {}'.format(
            batch_idx
        ))
        plt.savefig(os.path.join(opt.save_folder, 'fiedlervector_{}.png'.format(batch_idx)))
        plt.close(fig)

    return -1, L, U  # Haven't figure out how to get nc



def get_nc(class_hvs, pair_simil, thres, batch_idx, opt, warmup_done):
    """
    Obtain the number of clusters by searching for plateaus
    in the sorted eigenvalues

    Args:
        class_hvs: extracted class hypervectors by calling model.extract_class_hv()
        pair_simil: pairwise similarity between class hypervectors
        thres: the threshold for the pairwise similarity neighborhood
        batch_idx: the batch index in the training stream for logging
        opt: arguments
        warmup_done: whether warmup has been done

    Returns:
        nc: the number of clusters as the start of the plateau
        L: the k neighbors graph of the input class_hvs
        U: the eigenvectors of L
    """
    print('warmup done:', warmup_done)
    if not warmup_done:
        L = kneighbors_graph(class_hvs, 3, include_self=True).toarray()
    else:
        print('not warmup!!')
        L = (pair_simil > thres).astype('int')
    print(pair_simil)
    print(L)

    # plot_tsne_graph(class_hvs,
    #                fig_name=os.path.join(opt.save_folder,
    #                                      'cls_hv_{}.png'.format(batch_idx)))

    # Compute the eigenvalues and eigenvectors of L
    (S, U) = np.linalg.eig(L)
    S, U = np.real(S), np.real(U)
    ixs = np.argsort(-1 * S)  # Sort, descending
    S, U = S[ixs], U[:, ixs]
    U = U[:, S > 0]
    S = S[S > 0]
    S = S / S.max()

    # Find the first sorted eigenvalue that is 0.1 and use its index as nc
    #gap = S[:-1] - S[1:]
    #nc = np.argmax(gap) + 1
    nc = np.argmax(S < 0.1)

    print('Idx: {} nc={}'.format(
        batch_idx, nc
    ))

    # Plot sorted eigenvalues
    fig = plt.figure()
    plt.plot(np.arange(S.size), S)
    plt.title('Idx: {} nc={}'.format(
        batch_idx, nc
    ))
    plt.savefig(os.path.join(opt.save_folder, 'eigenvalue_{}.png'.format(batch_idx)))
    plt.close(fig)

    return nc, L, U


class LifeHDsemi():
    def __init__(self, opt, train_loader, val_loader,
                 num_classes, model, logger, device):
        self.opt = opt

        self.device = device

        # build data loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes

        # build model and criterion
        self.model = model

        # tensorboard
        self.logger = logger

        # warmup status
        self.warmup_done = False

    def start(self):
        for epoch in range(1, self.opt.epochs + 1):
            # train for one epoch
            time1 = time.time() 
            self.train(epoch)

            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # final validation
            acc = self.validate(epoch, len(self.train_loader), True, 'final')
            print('Stream final acc: {}'.format(acc))

    def warmup(self, idx, sample_hv, label):
        # warmup training
        if idx == 0:
            self.warmup_hvs = sample_hv
            self.warmup_labels = label

        elif idx < self.opt.warmup_batches:
            self.warmup_hvs = torch.cat((self.warmup_hvs, sample_hv), dim=0)
            self.warmup_labels = torch.cat((self.warmup_labels, label))
            # print(self.warmup_hvs.shape)  # (idx, D)
            # print(self.warmup_labels.shape)  # (idx)

        elif idx >= self.opt.warmup_batches:
            print(self.warmup_hvs)
            # End of the warmup session
            # Determine the number of clusters, and run spectral clustering to create init clusters
            nc, L, U = get_nc(self.warmup_hvs.cpu().numpy(), None, None,
                              idx, self.opt, self.warmup_done)
            # if nc is not valid, start with half the number of classes as default
            nc = nc if 0 < nc < self.model.max_classes - self.model.num_classes \
                else min(int(0.5 * self.model.max_classes), self.model.max_classes - self.model.num_classes)

            K2 = KMeans(nc)
            K2.fit(U[:, :nc])

            # Init cluster
            base = self.model.num_classes
            for ix in range(nc):
                cluster_mask = (K2.labels_ == ix)
                # print(i, np.sum(cluster_mask))
                self.model.classify_weights[base+ix] = self.warmup_hvs[cluster_mask].sum(dim=0)  # size 1xD
                self.model.classify_sample_cnt[base+ix] = cluster_mask.sum()

                # Only update mean and std when there are more than 1 sample in the init cluster
                #if cluster_mask.sum() > 1:
                dist_to_cen = F.normalize(self.warmup_hvs[cluster_mask]) @ \
                    F.normalize(self.model.classify_weights[ix].view(1, -1)).T
                    # Should be size sample_cntx1
                self.model.dist_mean[base+ix] = torch.mean(dist_to_cen)  # scalar
                self.model.dist_std[base+ix] = torch.mean(torch.abs(dist_to_cen - self.model.dist_mean[ix]))  # scalar

                self.model.last_edit[base+ix] = idx

            self.model.cur_classes = base + nc

            print('init # of clusters after warmup: {}'.format(nc))
            plot_tsne(self.warmup_hvs.cpu().numpy(),
                      K2.labels_, self.warmup_labels.cpu().numpy(),
                      title='warmup spectral clustering',
                      fig_name=os.path.join(self.opt.save_folder, 'warmup.png'))

            del self.warmup_hvs
            del self.warmup_labels
            self.warmup_done = True

    def train(self, epoch):
        """Training of one epoch on single-pass of data"""
        """Unsupervised method. Should not use the labels"""
        # Set validation frequency
        val_freq = np.floor(len(self.train_loader) / VAL_CNT).astype('int')
        merge_freq = np.floor(len(self.train_loader) / MERGE_CNT).astype('int')

        with torch.no_grad():
            cur_class = -1
            for idx, (images, labels) in enumerate(self.train_loader):

                labeled_idx = np.random.rand(images.shape[0]) < self.opt.label_ratio
                
                if labels[0] > cur_class:
                    class_shift.append(idx)
                    cur_class = labels[0]

                # Validation
                if idx > self.opt.warmup_batches and idx % val_freq == 0:
                    # Trick: trim the clusters that have samples less than 10
                    self.trim_clusters()

                    # acc, purity = self.validate(epoch, idx, False, 'before')
                    #################################################
                    # 3. Cluster merging
                    #################################################
                    pair_simil, class_hvs = self.model.extract_pair_simil()  # numpy array
                    if self.opt.dataset == 'har_timeseries' or \
                        self.opt.dataset == 'mhealth':
                        thres = self.model.dist_mean[:self.model.cur_classes].mean().cpu().numpy()
                    else:
                        thres = self.model.dist_mean[:self.model.cur_classes].max().cpu().numpy()
                    #print(self.model.dist_mean[:self.model.cur_classes])
                    #print('thres: ', thres)
                    nc, _, U = get_nc(class_hvs, pair_simil, thres,
                                      idx, self.opt, self.warmup_done)

                    # Merge clusters
                    if self.opt.k_merge_min < nc < self.model.max_classes:
                        self.merge_clusters(U, nc, class_hvs, idx)

                    acc, purity = self.validate(epoch, idx+1, False, 'after')
                    print('Validate stream: [{}][{}/{}]\tacc: {} purity: {}'.format(
                        epoch, idx + 1, len(self.train_loader), acc, purity))
                    sys.stdout.flush()

                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs, samples_hv = self.model(images)
                # print(sample_hv.shape)  # (batch_size, D)
                # print(outputs)

                # Supervised training using labeled samples
                for i in range(images.shape[0]):
                    if labeled_idx[i]:
                        self.add_sample_hv_to_exist_class(samples_hv[i],
                                                          labels[i],
                                                          outputs[i],
                                                          idx)

                # Separate labeled samples and unlabeld samples
                unlabeled_samples_hv = samples_hv[~labeled_idx]
                unlabeled_labels = labels[~labeled_idx]
                unlabeled_outputs = outputs[~labeled_idx]

                # Check if warmup has ended
                if not self.warmup_done:
                    self.warmup(idx, unlabeled_samples_hv, unlabeled_labels)

                else:
                    # Normal session after warmup
                    #################################################
                    # 1. predict the nearest centroid
                    #################################################
                    simil_to_class, pred_class = torch.max(unlabeled_outputs, dim=-1)
                    pred_class_samples = self.model.classify_sample_cnt[pred_class]

                    #print(
                    #    '\n\nidx: {}/{}\ncur_label: {}\nmin_dist: {}\npred_class: {}'.format(
                    #        idx, len(self.train_loader),
                    #        label.cpu().numpy(),
                    #        simil_to_class.cpu().numpy(), 
                    #        pred_class.cpu().numpy()))

                    assert pred_class_samples.min() > 0, \
                        'Predicted class {} has zero sample!'

                    #################################################
                    # 2. add sample to cluster or novelty detection
                    #################################################
                    # Novelty detection
                    # Compare the new max cosine similarity with the 95-percentile
                    # (given by opt.beta, mean - 3 * standard difference)
                    # in the pred_class's distance distribution
                    simil_threshold = self.model.dist_mean[pred_class] - \
                                        self.opt.beta * self.model.dist_std[pred_class]
                    #print('\tmean {}\n\tstd {}\n\tdist_thres {}'.format(
                    #    self.model.dist_mean[pred_class].cpu().numpy(),
                    #    self.model.dist_std[pred_class].cpu().numpy(),
                    #    simil_threshold.cpu().numpy()))
                    # print('threshold: ', simil_threshold.cpu().numpy())

                    # To show as a novelty, we require the samples in the existing cluster
                    # is larger than a fixed number (default is 10), so we have sufficient
                    # confidence
                    novel_detect_mask = (simil_to_class < simil_threshold) & \
                          (pred_class_samples > 10)  # (batch_size, D)
                    #print(simil_to_class < simil_threshold)
                    #print(pred_class_samples > 10)
                    #print(novel_detect_mask)

                    # Add the new sample to the predicted class
                    sorted_output, _ = torch.sort(unlabeled_outputs, dim=1, descending=True)
                    confidence = sorted_output[:, 0] - sorted_output[:, 1]
                    update_mask = confidence > self.opt.confidence
                    self.add_sample_hv_to_exist_class(unlabeled_samples_hv[update_mask],
                                                      pred_class[update_mask],
                                                      simil_to_class[update_mask],
                                                      idx)

                    # A novelty is detected, need to create new classes
                    if novel_detect_mask.sum() > 0:
                        #print('Novelty detected!')
                        novelty_detect.append(idx)
                        #print('pred class ', pred_class[novel_detect_mask].cpu().numpy())
                        #print('simil to class ', simil_to_class[novel_detect_mask].cpu().numpy())
                        #print('simil threshold ', simil_threshold[novel_detect_mask].cpu().numpy())
                        self.add_sample_hv_to_novel_class(unlabeled_samples_hv[novel_detect_mask], idx)

                self.model.classify.weight[:] = F.normalize(self.model.classify_weights)

                #print('sample cnt', self.model.classify_sample_cnt[:self.model.cur_classes].cpu().numpy().astype('int'))
                #print('mean', self.model.dist_mean[:self.model.cur_classes].cpu().numpy())
                #print('std', self.model.dist_std[:self.model.cur_classes].cpu().numpy())
            
            self.trim_clusters()
            print(self.model.classify_sample_cnt)
            plot_novelty_detection(class_shift, novelty_detect, self.opt.save_folder)
            

    def validate(self, epoch, loader_idx, plot, mode):  # task_list
        """Validation, evaluate linear classification accuracy and kNN accuracy"""
        test_samples, test_embeddings = None, None
        pred_labels, test_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Testing"):

                images = images.to(self.device)

                # compute classification accuracy
                outputs, _ = self.model(images)
                predictions = torch.argmax(outputs, dim=-1)

                # gather prediction results
                pred_labels += predictions.detach().cpu().tolist()
                test_labels += labels.cpu().tolist()

                # gather raw sampels and unnormalized embeddings
                embeddings = self.model.encode(images).detach().cpu().numpy()
                test_bsz = images.shape[0]
                if test_embeddings is None:
                    test_samples = images.squeeze().view(
                        (test_bsz, -1)).cpu().numpy()
                    test_embeddings = embeddings
                else:
                    test_samples = np.concatenate(
                        (test_samples,
                         images.squeeze().view((test_bsz, -1)).cpu().numpy()),
                        axis=0)
                    test_embeddings = np.concatenate(
                        (test_embeddings, embeddings),
                        axis=0)

        # log accuracy
        pred_labels = np.array(pred_labels).astype(int)
        print(np.unique(pred_labels))
        test_labels = np.array(test_labels).astype(int)
        acc, purity, cm = eval_acc(test_labels, pred_labels)
        print('Acc: {}, purity: {}'.format(acc, purity))

        nmi = eval_nmi(test_labels, pred_labels)
        print('NMI: {}'.format(nmi))

        ri = eval_ri(test_labels, pred_labels)
        print('RI: {}'.format(ri))

        with open(os.path.join(self.opt.save_folder, 'result.txt'), 'a+') as f:
            f.write('{epoch},{idx},{acc},{purity},{nmi},{ri}\n'.format(
                epoch=epoch, idx=loader_idx, acc=acc, purity=purity,
                nmi=nmi, ri=ri
            ))

        # tensorboard logger
        self.logger.log_value('accuracy', acc, loader_idx)
        self.logger.log_value('purity', purity, loader_idx)
        self.logger.log_value('nmi', nmi, loader_idx)
        self.logger.log_value('ri', ri, loader_idx)
        self.logger.log_value('num of clusters', self.model.cur_classes, loader_idx)

        # plot raw and high-dimensional embeddings
        if plot:
            # plot the tSNE of raw samples with predicted labels
            #plot_tsne(test_samples, np.array(pred_labels), np.array(test_labels),
            #          title='raw samples {} {} {}'.format(self.opt.method, self.opt.dataset, acc),
            #          fig_name=os.path.join(self.opt.save_folder,
            #                                '{}_sap_{}_{}.png'.format(
            #                                    loader_idx, self.opt.method, self.opt.dataset)))
            # plot the tSNE of embeddings with predicted labels
            plot_tsne(test_embeddings, np.array(pred_labels), np.array(test_labels),
                      title='embeddings {} {} {} {}'.format(self.opt.method, self.opt.dataset, acc, mode),
                      fig_name=os.path.join(self.opt.save_folder,
                                            '{}_emb_{}_{}_{}.png'.format(
                                                loader_idx, self.opt.method, self.opt.dataset, mode)))

            # plot embeddings with class hypervectors
            #class_hvs = self.model.extract_class_hv()  # numpy array
            #plot_tsne_graph(class_hvs,
            #                title='class hvs {} {} {}'.format(self.opt.method, self.opt.dataset, acc),
            #                fig_name=os.path.join(self.opt.save_folder,
            #                                      '{}_cls_hv_{}_{}.png'.format(
            #                                          loader_idx, self.opt.method, self.opt.dataset)))

            # save confusion matrix
            np.save(os.path.join(self.opt.save_folder, 'confusion_mat'), cm)

        return acc, purity

    def add_sample_hv_to_exist_class(self, sample_hv, pred_class, simil_to_class, batch_idx):
        """
        Use new sample_hv to update the predicted class in model

        Args:
            sample_hv: tensor, the hypevectors of the new samples in shape (batch_size, D)
            pred_class: tensor, the predicted class of sample_hv in shape (batch_size,)
            simil_to_class: tensor, cosine similarity with the predicted class
            batch_idx: the batch index in the training stream for logging
        """
        pred_class_set = np.unique(pred_class.cpu().numpy())
        # print(pred_class_set)

        for cs in pred_class_set:
            mask = (pred_class == cs)
            old_sample_hv = copy.deepcopy(self.model.classify_weights[cs].view(1, -1))
            old_sample_cnt = self.model.classify_sample_cnt[cs].item()
            self.model.classify_weights[cs] += sample_hv[mask].sum(dim=0)
            self.model.classify_sample_cnt[cs] += mask.sum()
            # print(old_sample_cnt)

            if old_sample_cnt > 1: # Update the mean and std, which are scalar
                self.model.dist_std[cs] = \
                    self.opt.alpha * torch.mean(torch.abs(simil_to_class[mask] - self.model.dist_mean[cs])) + \
                    (1 - self.opt.alpha) * self.model.dist_std[cs]
                self.model.dist_mean[cs] = \
                    self.opt.alpha * simil_to_class[mask].mean() + \
                    (1 - self.opt.alpha) * self.model.dist_mean[cs]

            elif old_sample_cnt == 1:  # Assign mean and std for the first time when there is only one sample in class
                dist_to_cen = \
                    F.normalize(torch.cat((old_sample_hv, sample_hv[mask]), dim=0)) @ \
                    F.normalize(self.model.classify_weights[cs].view(1, -1)).T
                    # Should be size (sample_cnt+1)x1
                self.model.dist_mean[cs] = torch.mean(dist_to_cen)  # scalar
                self.model.dist_std[cs] = torch.mean(
                    torch.abs(dist_to_cen - self.model.dist_mean[cs]))  # scalar

            else:  # This is the very first sample for this class
                pass

            self.model.last_edit[cs] = batch_idx

    def add_sample_hv_to_novel_class(self, sample_hv, batch_idx):
        """
        Use new sample_hv to create a novel class after detecting novelty

        Args:
            sample_hv: tensor, the hypevectors of the new samples in shape (batch_size, D)
            batch_idx: the batch index in the training stream for logging
        """
        unsupervised_mask = np.zeros(self.model.max_classes)
        unsupervised_mask[self.model.num_classes:] = 1
        unsupervised_mask = unsupervised_mask.astype(bool)
        if self.model.cur_classes < self.model.max_classes:
            # Simply create a new class and do not update the mean and std right now
            new_cs = self.model.cur_classes
            self.model.cur_classes += 1

        else:
            # The current number of classes has reached max classes
            # Find the least-used class to replace
            assert np.min(self.model.last_edit[unsupervised_mask]) > 0, \
                'not all unsupervised classes are edited!'

            new_cs = np.argmin(self.model.last_edit[unsupervised_mask]).astype('int')
            new_cs = new_cs + self.model.num_classes  # Add the supervised class base

        # print(new_cs)
        self.model.classify_weights[new_cs] = sample_hv.sum(dim=0)
        self.model.classify_sample_cnt[new_cs] = sample_hv.shape[0]
        self.model.last_edit[new_cs] = batch_idx

        if sample_hv.shape[0] > 1:  # more than one sample
            dist_to_cen = F.normalize(sample_hv) @ \
                F.normalize(self.model.classify_weights[new_cs].view(1, -1)).T
                # Should be size sample_cntx1
            self.model.dist_mean[new_cs] = torch.mean(dist_to_cen)  # scalar
            self.model.dist_std[new_cs] = torch.mean(
                torch.abs(dist_to_cen - self.model.dist_mean[new_cs])
                )  # scalar

    def merge_clusters(self, U, nc, class_hvs, batch_idx):
        """
        Merge existing clusters (classes) in the HDC model into new clusters with new_labels

        Args:
            U: matrix of eigenvectors
            nc: int, number of clusters
            class_hvs: numpy array, current class hypervectors, size (class #, HD dim)
            batch_idx: the batch index in the training stream for logging
        """
        K2 = KMeans(nc)
        K2.fit(U[:, :nc])

        plot_tsne_graph(class_hvs,
                        title='before: {}'.format(K2.labels_),
                        fig_name=os.path.join(self.opt.save_folder, '{}_before.png'.format(batch_idx)))

        old_classes = self.model.cur_classes
        old_weights = copy.deepcopy(self.model.classify_weights[:old_classes])
        old_cnt = copy.deepcopy(self.model.classify_sample_cnt[:old_classes])
        old_dist_mean = copy.deepcopy(self.model.dist_mean[:old_classes])
        old_dist_std = copy.deepcopy(self.model.dist_std[:old_classes])
        old_last_edit = copy.deepcopy(self.model.last_edit[:old_classes])

        self.model.classify.weight.data.fill_(0.0)
        self.model.classify_weights = copy.deepcopy(self.model.classify.weight)
        self.model.classify_sample_cnt = torch.zeros(self.model.max_classes).to(self.device)
        self.model.dist_mean = torch.zeros(self.model.max_classes).to(self.device)
        self.model.dist_std = torch.zeros(self.model.max_classes).to(self.device)
        self.model.last_edit = - np.ones(self.model.max_classes)

        # Clean K2 labels to remove the non-assigned clusters
        sorted_labels = sorted(list(set(K2.labels_)))
        new_nc = len(sorted_labels)
        self.model.cur_classes = new_nc
        new_labels = np.array([sorted_labels.index(K2.labels_[i])
                               for i in range(len(K2.labels_))])

        # Fill in the information of the combined new clusters
        print('K2 labels: ', K2.labels_)
        print('new labels: ', new_labels)
        print('Starting merging: {}->{}'.format(len(K2.labels_), new_nc))

        # Update label class hypervector first
        new_labels_valid = np.ones(new_nc).astype(bool)
        for ix in range(self.model.num_classes):
            new_label = new_labels[ix]
            old_cluster_mask = (new_labels == new_label)
            # print(ix, old_cluster_mask)
            if old_cluster_mask[:self.model.num_classes].sum() > 1:
                # Attempt to merge to labeled class hypervector, abort
                self.model.classify_weights[ix] = old_weights[ix]
                self.model.classify_sample_cnt[ix] = old_cnt[ix]
                self.model.dist_mean[ix] = old_dist_mean[ix]
                self.model.dist_std[ix] = old_dist_std[ix]
                self.model.last_edit = old_last_edit[ix]
                new_labels_valid[new_label] = False  # new label is invalid

            else:
                self.model.classify_weights[ix] = old_weights[old_cluster_mask].sum(dim=0)  # size 1xD
                self.model.classify_sample_cnt[ix] = old_cnt[old_cluster_mask].sum()
                self.model.dist_mean[ix] = torch.max(old_dist_mean[old_cluster_mask])
                self.model.dist_std[ix] = torch.max(old_dist_std[old_cluster_mask])

                self.model.last_edit[ix] = np.max(old_last_edit[old_cluster_mask])

        # Clean new labels to remove the invalid labels

        sorted_labels = sorted(np.arange(new_nc)[new_labels_valid])
        new_nc = len(sorted_labels)
        self.model.cur_classes = new_nc
        new_labels = np.array([sorted_labels.index(new_labels[i])
                               for i in range(len(new_labels))])

        for ix in range(self.model.num_classes, new_nc):

            old_cluster_mask = (new_labels == ix)
            # print(ix, old_cluster_mask)

            self.model.classify_weights[ix] = old_weights[old_cluster_mask].sum(dim=0)  # size 1xD
            self.model.classify_sample_cnt[ix] = old_cnt[old_cluster_mask].sum()
            print('{} old clusters {}\n\twith cnt {}'.format(
                old_cluster_mask.sum(),
                np.arange(len(K2.labels_))[old_cluster_mask],
                old_cnt[old_cluster_mask]
            ))

            self.model.dist_mean[ix] = torch.max(old_dist_mean[old_cluster_mask])
            self.model.dist_std[ix] = torch.max(old_dist_std[old_cluster_mask])
            self.model.last_edit[ix] = np.max(old_last_edit[old_cluster_mask])

        self.model.classify.weight[:] = F.normalize(self.model.classify_weights)

        new_class_hvs = self.model.extract_class_hv()  # numpy array
        plot_tsne_graph(new_class_hvs,
                        title='after: {}'.format(np.arange(self.model.cur_classes)),
                        fig_name=os.path.join(self.opt.save_folder, '{}_after.png'.format(batch_idx)))
        
    def trim_clusters(self):
        """
        Trim clusters without ground-truth labels with less than 10 sample counts.
        """
        supervised_mask = np.zeros(self.model.max_classes)
        supervised_mask[:self.model.num_classes] = 1
        mask_to_keep = self.model.classify_sample_cnt.cpu().numpy() > 10
        mask_to_keep = mask_to_keep | supervised_mask.astype(bool)
        new_classes = mask_to_keep.sum()
        
        print('\n\nTrim!!! {}->{}'.format(self.model.cur_classes, new_classes))

        old_weights = copy.deepcopy(self.model.classify_weights)
        old_cnt = copy.deepcopy(self.model.classify_sample_cnt)
        old_dist_mean = copy.deepcopy(self.model.dist_mean)
        old_dist_std = copy.deepcopy(self.model.dist_std)
        old_last_edit = copy.deepcopy(self.model.last_edit)

        self.model.cur_classes = new_classes

        self.model.classify.weight.data.fill_(0.0)
        self.model.classify_weights = copy.deepcopy(self.model.classify.weight)
        self.model.classify_sample_cnt = torch.zeros(self.model.max_classes).to(self.device)
        self.model.dist_mean = torch.zeros(self.model.max_classes).to(self.device)
        self.model.dist_std = torch.zeros(self.model.max_classes).to(self.device)
        self.model.last_edit = - np.ones(self.model.max_classes)

        self.model.classify_weights[:new_classes] = old_weights[mask_to_keep]
        self.model.classify_sample_cnt[:new_classes] = old_cnt[mask_to_keep]
        self.model.dist_mean[:new_classes] = old_dist_mean[mask_to_keep]
        self.model.dist_std[:new_classes] = old_dist_std[mask_to_keep]
        self.model.last_edit[:new_classes] = old_last_edit[mask_to_keep]

        self.model.classify.weight[:] = F.normalize(self.model.classify_weights)