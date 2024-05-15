from __future__ import print_function

import os
import copy
import numpy as np
import sys
import argparse
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.transforms.functional import rotate
from utils.eval_utils import eval_acc, eval_nmi, eval_ri
from utils.plot_utils import plot_tsne, plot_select_tsne, plot_pca

VAL_CNT = 10

class BasicHD():
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


    def start(self):
        for epoch in range(1, self.opt.epochs + 1):
            # train for one epoch
            time1 = time.time()
            self.train(self.train_loader, self.val_loader, self.model,
                  epoch, self.opt, self.logger)

            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # final validation
            acc = self.validate(self.val_loader, self.model,
                       epoch, len(self.train_loader), self.opt, self.logger, self.opt.plot)
            print('Stream final acc: {}'.format(acc))


    def train(self,train_loader, val_loader, model, epoch, opt, logger):  # task_list
        """Training of one epoch on single-pass of data"""
        # Set validation frequency
        val_freq = np.floor(len(train_loader) / VAL_CNT).astype('int')
        batchs_per_class = np.floor(len(train_loader) / self.num_classes).astype('int')

        with torch.no_grad():
            # for idx, (images, labels) in enumerate(train_loader):
            idx = 0  # batch index
            class_batch_idx = 0  # batch index in the current class
            cur_class = -1
            self.mask = None

            for images, labels in tqdm(train_loader, desc="Training"):
                # print(labels.detach().cpu().tolist())

                if idx > 0 and idx % val_freq == 0:
                    acc = self.validate(val_loader, model, epoch, idx, opt, logger, False)
                    print('Validate stream: [{}][{}/{}]\tacc: {}'.format(
                        epoch, idx + 1, len(train_loader), acc))
                    sys.stdout.flush()

                    # Figure out the mask depending on the model.classify.weight
                    weight_sum = torch.abs(self.model.classify_weights.sum(dim=0))
                    sort_idx = torch.argsort(weight_sum, descending=True)
                    self.mask = torch.zeros(self.opt.dim, device=self.device).type(torch.bool)
                    self.mask[sort_idx[:self.opt.mask_dim]] = 1
                    self.cur_mask_dim = self.opt.mask_dim

                if labels[0] > cur_class:
                    class_batch_idx = 0
                    cur_class = labels[0]
                
                if self.opt.rotation > 0.0:
                    rot_degrees = self.opt.rotation / batchs_per_class * class_batch_idx
                    images = rotate(images, rot_degrees)

                images = images.to(self.device)
                labels = labels.to(self.device)

                samples_hv = model.encode(images, self.mask)

                for i in range(images.shape[0]):
                    model.classify_weights[labels[i]] += samples_hv[i]
                    model.classify_sample_cnt[labels[i]] += 1

                model.classify.weight[:] = F.normalize(model.classify_weights)

                idx += 1
                class_batch_idx += 1


    def validate(self, val_loader, model, epoch, loader_idx,
                 opt, logger, plot):  # task_list
        """Validation, evaluate linear classification accuracy and kNN accuracy"""
        test_samples, test_embeddings = None, None
        pred_labels, test_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Testing"):

                images = images.to(self.device)

                outputs, _ = model(images)
                #print('outputs', outputs)

                predictions = torch.argmax(outputs, dim=-1)
                #print('pred', predictions)
                #print('test label', labels)

                # gather prediction results
                pred_labels += predictions.detach().cpu().tolist()
                test_labels += labels.detach().cpu().tolist()

                # gather raw sampels and unnormalized embeddings
                embeddings = model.encode(images).detach().cpu().numpy()
                test_bsz = images.shape[0]
                if test_embeddings is None:
                    test_samples = images.squeeze().view((test_bsz, -1)).cpu().numpy()
                    test_embeddings = embeddings
                else:
                    test_samples = np.concatenate((test_samples,
                                                   images.squeeze().view((test_bsz, -1)).cpu().numpy()),
                                                  axis=0)
                    test_embeddings = np.concatenate((test_embeddings, embeddings),
                                                     axis=0)

        # log accuracy
        pred_labels = np.array(pred_labels).astype(int)
        test_labels = np.array(test_labels).astype(int)
        acc = np.sum(pred_labels == test_labels) / pred_labels.size
        print('Acc: {}'.format(acc))

        acc_pc, num_pc = np.zeros(self.num_classes), np.zeros(self.num_classes)
        for i in range(self.num_classes):
            class_mask = (test_labels == i)
            acc_pc[i] = (pred_labels[class_mask] == test_labels[class_mask]).sum() / \
                class_mask.sum()
            num_pc[i] = class_mask.sum()
        print('Acc per class: {}'.format(acc_pc))
        print('Samples per class: {}'.format(num_pc))
        print('Avg acc per class: {}'.format(acc_pc.mean()))

        nmi = eval_nmi(test_labels, pred_labels)
        print('NMI: {}'.format(nmi))

        ri = eval_ri(test_labels, pred_labels)
        print('RI: {}'.format(ri))

        with open(os.path.join(self.opt.save_folder, 'result.txt'), 'a+') as f:
            f.write('{epoch},{idx},{acc},{nmi},{ri}\n'.format(
                epoch=epoch, idx=loader_idx, acc=acc,
                nmi=nmi, ri=ri
            ))

        # tensorboard logger
        logger.log_value('accuracy', acc, loader_idx)
        logger.log_value('nmi', nmi, loader_idx)
        logger.log_value('ri', ri, loader_idx)

        # plot raw and high-dimensional embeddings
        if plot:
            # plot the tSNE of raw testing
            plot_tsne(test_samples, np.array(pred_labels), np.array(test_labels),
                      title='raw samples {} {} {}'.format(opt.method, opt.dataset, acc),
                      fig_name=os.path.join(opt.save_folder,
                                            'sap_{}_{}.png'.format(
                                                opt.method, opt.dataset)))
            plot_tsne(test_embeddings, np.array(pred_labels), np.array(test_labels),
                      title='embeddings {} {} {}'.format(opt.method, opt.dataset, acc),
                      fig_name=os.path.join(opt.save_folder,
                                            'tsne_emb_{}_{}.png'.format(
                                                opt.method, opt.dataset)))
            plot_pca(test_embeddings, np.array(pred_labels), np.array(test_labels),
                     title='embeddings {} {} {}'.format(opt.method, opt.dataset, acc),
                     fig_name=os.path.join(opt.save_folder,
                                           'pca_emb_{}_{}.png'.format(
                                                opt.method, opt.dataset)))

            # add class hypervectors
            class_hvs = model.extract_class_hv()  # numpy array
            test_embeddings = np.concatenate((test_embeddings, class_hvs),
                                             axis=0)
            test_labels = np.concatenate(
                (test_labels, np.arange(model.num_classes)),
                axis=0)
            select_indexes = list(range(-model.num_classes, 0))
            plot_select_tsne(test_embeddings, test_labels, select_indexes,
                             title='{} {} {}'.format(opt.method, opt.dataset, acc),
                             fig_name=os.path.join(opt.save_folder,
                                                   'emb_{}_{}_cen.png'.format(
                                                       opt.method, opt.dataset)))

        return acc
