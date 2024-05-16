import numpy as np
import random
import torch
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets

from .dataset.har import HAR
from .dataset.har_timeseries import HAR_TimeSeries
from .dataset.isolet import ISOLET
from .dataset.mhealth import MHEALTH
from .dataset.esc50 import ESC50

#cifar100_class_array = np.array([4,6,17])  # Use in plotting tsne
cifar100_class_array = np.arange(20)

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.'
    Code copied from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


class SeqSampler(Sampler):
    def __init__(self, dataset_name, dataset, blend_ratio, n_concurrent_classes,
                 imbalanced, train_samples_ratio):
        """data_source is a Subset"""
        self.dataset_name = dataset_name
        self.num_samples = len(dataset)
        self.blend_ratio = blend_ratio
        self.n_concurrent_classes = n_concurrent_classes
        self.imbalanced = imbalanced
        self.train_samples_ratio = train_samples_ratio
        self.total_sample_num = int(self.num_samples * train_samples_ratio)

        # Configure the correct train_subset and val_subset
        if torch.is_tensor(dataset.targets):
            self.labels = dataset.targets.detach().cpu().numpy()
        else:  # targets in cifar10 and cifar100 is a list
            self.labels = np.array(dataset.targets)
        self.classes = list(set(self.labels))
        self.n_classes = len(self.classes)

    def __iter__(self):
        """Sequential sampler"""
        # Configure concurrent classes
        cmin = []
        cmax = []
        for i in range(int(self.n_classes / self.n_concurrent_classes)):
                cmin.append(i * self.n_concurrent_classes)
                cmax.append((i + 1) * self.n_concurrent_classes)
        print('cmin', cmin)
        print('cmax', cmax)

        filter_fn = lambda y: np.logical_and(
            np.greater_equal(y, cmin[c]), np.less(y, cmax[c]))

        # Configure sequential class-incremental input
        sample_idx = []
        for c in range(int(self.n_classes / self.n_concurrent_classes)):
            filtered_train_ind = filter_fn(self.labels)
            filtered_ind = np.arange(self.labels.shape[0])[filtered_train_ind]
            np.random.shuffle(filtered_ind)

            # The sample num should be scaled according to train_samples_ratio
            cls_sample_num = int(filtered_ind.size * self.train_samples_ratio)

            if self.imbalanced:  # Imbalanced class
                cls_sample_num = int(cls_sample_num * np.random.uniform(low=0.5, high=1.0))

            sample_idx.append(filtered_ind.tolist()[:cls_sample_num])
            print('Class [{}, {}): {} samples'.format(cmin[c], cmax[c], cls_sample_num))

        # Configure blending class
        if self.blend_ratio > 0.0:
            for c in range(len(self.classes)):
                # Blend examples from the previous class if not the first
                if c > 0:
                    blendable_sample_num = \
                        int(min(len(sample_idx[c]), len(sample_idx[c-1])) * self.blend_ratio / 2)
                    # Generate a gradual blend probability
                    blend_prob = np.arange(0.5, 0.05, -0.45 / blendable_sample_num)
                    assert blend_prob.size == blendable_sample_num, \
                        'unmatched sample and probability count'

                    # Exchange with the samples from the end of the previous
                    # class if satisfying the probability, which decays
                    # gradually
                    for ind in range(blendable_sample_num):
                        if random.random() < blend_prob[ind]:
                            tmp = sample_idx[c-1][-ind-1]
                            sample_idx[c-1][-ind-1] = sample_idx[c][ind]
                            sample_idx[c][ind] = tmp

        final_idx = []
        for sample in sample_idx:
            final_idx += sample

        # Update total sample num
        self.total_sample_num = len(final_idx)

        return iter(final_idx)

    def __len__(self):
        return self.total_sample_num

def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'mnist':
        mean = (0.1307,)
        std = (0.3081,)
    elif opt.dataset == 'har' or \
            opt.dataset == 'har_timeseries' or \
            opt.dataset == 'isolet' or \
            opt.dataset == 'mhealth' or \
            opt.dataset == 'esc50':
        # normalization is completed during loading
        mean = (0.0,)
        std = (1.0,)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    transform_image = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,  # Note, CIFAR-10, CIFAR-100 and TinyImageNet needs normalization
                    # before the resnet backbone
                    # MNIST uses random projection. Using normalization or not does not
                    # have any different on the accuracy
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=transform_image,
                                         download=True,
                                         train=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=transform_image)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=transform_image,
                                          download=True,
                                          train=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=transform_image)

        # Convert sparse labels to coarse labels
        train_dataset.targets = sparse2coarse(train_dataset.targets)
        val_dataset.targets = sparse2coarse(val_dataset.targets)

        labels_np = val_dataset.targets
        mask = np.isin(labels_np, cifar100_class_array)
        val_dataset.data = val_dataset.data[mask]
        val_dataset.targets = val_dataset.targets[mask]

    elif opt.dataset == 'har':
        train_dataset = HAR(root=opt.data_folder + 'HAR',
                            transform=transform,
                            train=True,
                            download=True)
        val_dataset = HAR(root=opt.data_folder + 'HAR',
                          train=False,
                          transform=transform)

    elif opt.dataset == 'har_timeseries':
        test_partition = opt.test_samples_ratio / \
                         (opt.train_samples_ratio + opt.test_samples_ratio)
        train_dataset = HAR_TimeSeries(root=opt.data_folder + 'HAR_TimeSeries',
                                       transform=transform,
                                       train=True,
                                       download=True,
                                       test_partition=test_partition,
                                       win_size=opt.win_size,
                                       overlap=opt.overlap)
        val_dataset = HAR_TimeSeries(root=opt.data_folder + 'HAR_TimeSeries',
                                     train=False,
                                     transform=transform,
                                     test_partition=test_partition,
                                     win_size=opt.win_size,
                                     overlap=opt.overlap)

    elif opt.dataset == 'mhealth':
        test_partition = opt.test_samples_ratio / \
                         (opt.train_samples_ratio + opt.test_samples_ratio)
        train_dataset = MHEALTH(root=opt.data_folder + 'MHEALTH',
                                transform=transform,
                                train=True,
                                download=True,
                                test_partition=test_partition,
                                win_size=opt.win_size,
                                overlap=opt.overlap)
        val_dataset = MHEALTH(root=opt.data_folder + 'MHEALTH',
                              train=False,
                              transform=transform,
                              test_partition=test_partition,
                              win_size=opt.win_size,
                              overlap=opt.overlap)

    elif opt.dataset == 'isolet':
        train_dataset = ISOLET(root=opt.data_folder + 'ISOLET',
                                        transform=transform,
                                        train=True,
                                        download=True)
        val_dataset = ISOLET(root=opt.data_folder + 'ISOLET',
                             train=False,
                             transform=transform)

    elif opt.dataset == 'esc50':
        train_dataset = ESC50(root=opt.data_folder + 'esc50',
                              transform=transform,
                              train=True,
                              download=True,
                              sr=opt.sampling_rate,
                              fold=opt.fold)
        val_dataset = ESC50(root=opt.data_folder + 'esc50',
                            train=False,
                            transform=transform,
                            sr=opt.sampling_rate,
                            fold=opt.fold)

    elif opt.dataset == 'mnist':
        train_dataset = datasets.MNIST(root=opt.data_folder,
                                       transform=transform_image,
                                       download=True,
                                       train=True)
        val_dataset = datasets.MNIST(root=opt.data_folder,
                                     train=False,
                                     transform=transform_image)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                             transform=transform)
        val_dataset = datasets.ImageFolder(root=opt.data_folder,
                                           transform=transform)
    else:
        raise ValueError(opt.dataset)

    # Create training loader
    if opt.training_data_type == 'iid':
        train_subset_len = int(len(train_dataset) * opt.train_samples_ratio)
        train_subset, _ = torch.utils.data.random_split(dataset=train_dataset,
                                                        lengths=[train_subset_len,
                                                                 len(train_dataset) - train_subset_len])
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=opt.batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=True, sampler=None)
    else:  # sequential
        if opt.dataset == 'har_timeseries' or opt.dataset == 'mhealth':
            # Keep the original order
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=opt.batch_size, shuffle=False,
                num_workers=opt.num_workers, pin_memory=True, sampler=None)
        else:
            # Sample the dataset using SeqSampler
            train_sampler = SeqSampler(opt.dataset,
                                    train_dataset,
                                    opt.blend_ratio,
                                    opt.n_concurrent_classes,
                                    opt.imbalanced,
                                    opt.train_samples_ratio)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=opt.batch_size, shuffle=False,
                num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    # Create validation loader
    test_subset_len = int(len(val_dataset) * opt.test_samples_ratio)
    test_subset, _ = torch.utils.data.random_split(dataset=val_dataset,
                                                  lengths=[test_subset_len,
                                                           len(val_dataset) - test_subset_len])
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=opt.val_batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

    # Obtain the total number of classes
    if torch.is_tensor(train_dataset.targets):
        labels = train_dataset.targets.detach().cpu().numpy()
    else:  # targets in cifar10 and cifar100 is a list
        labels = np.array(train_dataset.targets)
    labels = labels[labels != -1]
    num_labels = np.unique(labels).size
    print('Num of classes: ', num_labels)

    return train_loader, test_loader, num_labels
