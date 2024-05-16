import os
import sys
import argparse
import random
import numpy as np
import torch
import tensorboard_logger as tb_logger

from utils.set_utils import set_model
from utils.data_utils import set_loader

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--val_batch_size', type=int, default=16,
                        help='batch_size in validation')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num of workers to use')
    parser.add_argument('--steps_per_batch_stream', type=int, default=20,
                        help='number of steps for per batch of streaming data')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs or number of passes on dataset')

    # optimization
    parser.add_argument('--learning_rate_stream', type=float, default=0.1,
                        help='learning rate for streaming new data')
    # parser.add_argument('--lr_decay_epochs', type=str, default=' 00,800,900',
    #                    help='where to decay lr, can be a list')
    # parser.add_argument('--lr_decay_rate', type=float, default=0.1,
    #                    help='decay rate for learning rate')
    #parser.add_argument('--weight_decay', type=float, default=1e-4,
    #                    help='weight decay')
    #parser.add_argument('--momentum', type=float, default=0.9,
    #                    help='momentum')


    # model dataset
    parser.add_argument('--feature_ext', type=str, default='none',
                        choices=['none', 'resnet18', 'resnet50', 'acdnet', 
                                 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small'],
                        help='feature extractor to use on raw data')
    parser.add_argument('--feature_ext_ckpt', type=str, default='none',
                        help='path to the ckpt of feature extractor')
    parser.add_argument('--pretrained_on', type=str, default='none',
                        choices=['none', 'imagenet', 'cifar10'])
    parser.add_argument('--hd_encoder', type=str, default='none',
                        choices=['none', 'rp', 'idlevel', 'nonlinear', 'spatiotemporal'],
                        help='the type of hd encoding function to use')

    parser.add_argument('--dim', type=int, default=10000,
                        help='the size of HD space dimension')
    parser.add_argument('--mask_dim', type=int, default=10000,
                        help='the size of mask dimension for efficiency design')
    parser.add_argument('--mask_mode', type=str, default='fixed',
                        choices=['fixed', 'adaptive'],
                        help='the mode for adjusting the mask dimension')
    parser.add_argument('--num_levels', type=int, default=100,
                        help='the number of quantized level used on raw data')
    parser.add_argument('--randomness', type=float, default=0.5,
                        help='the randomness during generating level hypervectors')
    parser.add_argument('--flipping', type=float, default=0.01,
                        help='the flipping rate in the time series encoder')
    parser.add_argument('--win_size', type=int, default=128,
                        help='sliding window size for time-series data')
    parser.add_argument('--overlap', type=float, default=0.75,
                        help='the ratio of overlap in generating sliding window')
    

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist', 'har', 'har_timeseries', 'mhealth',
                                 'isolet', 'esc50'],
                        help='dataset')

    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=28, help='parameter for RandomResizedCrop')
    parser.add_argument('--training_data_type', type=str, default='iid',
                        choices=['iid', 'class_iid', 'instance', 'class_instance'],
                        help='iid or sequential datastream')
    parser.add_argument('--blend_ratio', type=float, default=.0,
                        help="the ratio blend classes at the boundary")
    parser.add_argument('--n_concurrent_classes', type=int, default=1,
                        help="the number of concurrent classes showing at the same time")
    parser.add_argument('--imbalanced', default=False, action="store_true",
                        help='whether the image stream is imbalanced')
    parser.add_argument('--train_samples_ratio', type=float, default=1.0,
                        help="the ratio of total training samples used in training")
    parser.add_argument('--test_samples_ratio', type=float, default=0.9,
                        help="the ratio of total testing samples used in testing")
    parser.add_argument('--label_ratio', type=float, default=0.1,
                        help="the ratio of labeled samples")
    parser.add_argument('--warmup_batches', type=int, default=4,
                        help='number of warmup batches')
    parser.add_argument('--rotation', type=float, default=0,
                        help='rotate degrees per incremental class')

    # for ESC-50
    parser.add_argument('--sampling_rate', type=int, default=20000,
                        help='sampling frequency in the esc-50 dataset')
    parser.add_argument('--fold', type=int, default=4,
                        help='which fold to use for testing')

    # method
    parser.add_argument('--method', type=str, default='hd',
                        choices=['BasicHD', 'LifeHD', 'SemiHD', 'LifeHDsemi'],
                        help='choose method')

    parser.add_argument('--max_classes', type=int, default=10,
                        help='max number of classes in unsupervised version')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature for scaling cosine similarity between '
                             'class hypervector')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='centroid learning rate')
    parser.add_argument('--beta', type=float, default=3.0,
                        help='standard difference multiplier for novelty detection')
    parser.add_argument('--merge_freq', type=int, default=25,
                        help='number of batches to fire the next merge')
    parser.add_argument('--hit_th', type=int, default=10,
                        help='threshold for trim')
    parser.add_argument('--k_merge_min', type=int, default=10,
                        help='number of nearest neighbors during drawing the knn graph')
    parser.add_argument('--merge_mode', type=str, default='merge',
                        choices=['merge', 'no_merge', 'no_trim'],
                        help='the mode for cluster merging')

    # Evaluation
    parser.add_argument('--k_scale', type=float, default=1.0,
                        help='to scale the number of classes during evaluation')
    parser.add_argument('--plot', default=False, action="store_true",
                        help="whether to plot during evaluation")

    parser.add_argument('--trial', type=int, default=0,
                        help='id for recording multiple runs')

    parser.add_argument('--confidence', type=float, default=0.07,
                        help='The confidence threshold to use unsampled data for classification')

    opt = parser.parse_args()

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/{}/{}_models/'.format(opt.method, opt.dataset)
    opt.tb_path = './save/{}/{}_tensorboard/'.format(opt.method, opt.dataset)

    opt.model_name = '{}_{}_{}_{}_dim{}_{}_{}_{}_{}_{}_data_{}_{}_{}_{}_{}_{}_' \
                     'lrs_{}_bsz_{}_mem_{}_{}_{}_{}_{}_{}_{}_epoch_{}_trial_{}'.format(
        opt.method, opt.dataset, opt.feature_ext, opt.hd_encoder,
        opt.dim, opt.mask_dim, opt.mask_mode,
        opt.num_levels, opt.randomness, opt.flipping, opt.training_data_type, 
        opt.blend_ratio, opt.n_concurrent_classes, int(opt.imbalanced), opt.rotation,
        opt.label_ratio, opt.learning_rate_stream, opt.batch_size, opt.max_classes, 
        opt.merge_freq, opt.merge_mode, opt.k_merge_min, 
        opt.alpha, opt.beta, opt.hit_th, opt.epochs, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt
    

def main():
    opt = parse_option()
    
    print("============================================")
    print(opt)
    print("============================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    # set seed for reproducing
    random.seed(opt.trial)
    np.random.seed(opt.trial)
    torch.manual_seed(opt.trial)

    # build data loader
    train_loader, val_loader, num_classes = set_loader(opt)

    # build model and criterion
    model = set_model(opt, num_classes, device)
    model.to(device)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    ######################################################
    # Supervised HD methods
    ######################################################
    if opt.method == "BasicHD":

        from methods.BasicHD.BasicHD import BasicHD
        Trainer = BasicHD(opt, train_loader, val_loader,
                          num_classes, model, logger, device)
        Trainer.start()

    ######################################################
    # Unsupervised HD methods
    ######################################################
    elif opt.method == "LifeHD":

        from methods.LifeHD.LifeHD import LifeHD
        Trainer = LifeHD(opt, train_loader, val_loader,
                         num_classes, model, logger, device)
        Trainer.start()

    ######################################################
    # SemiHD baseline
    ######################################################
    elif opt.method == "SemiHD":

        from methods.SemiHD.SemiHD import SemiHD
        Trainer = SemiHD(opt, train_loader, val_loader,
                         num_classes, model, logger, device)
        Trainer.start()

    ######################################################
    # Semi Unsupervised HD methods
    ######################################################
    elif opt.method == "LifeHDsemi":

        from methods.LifeHDsemi.LifeHDsemi import LifeHDsemi
        Trainer = LifeHDsemi(opt, train_loader, val_loader,
                             num_classes, model, logger, device)
        Trainer.start()

if __name__ == '__main__':
    main()
    