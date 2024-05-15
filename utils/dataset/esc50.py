import os
import sys
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from . import esc50_utils as U

class ESC50(Dataset):
    """
    Defines ESC-50 dataset.
    """

    def __init__(self, root: str, train: bool = True, transform: transforms = None,
                 target_transform: transforms = None, download: bool = False,
                 sr: int = 20000, nFolds: int = 5, nClasses: int = 50,
                 inputLength: int = 30225, fold: int = 4) -> None:
        """
        Extra arguments:
            sr: int, sampling rate
            nFolds: int, number of folds in total
            inputLength: int, length of the input after preprocessing
            fold: int, which fold to use for testing
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download  # NOT USED, already downloaded with prepare_esc50_training.py
        self.sr = sr
        self.nFolds = nFolds
        self.nClasses = nClasses
        self.inputLength = inputLength
        self.fold = fold
        self.preprocess_funcs = self.preprocess_setup()

        mode = 'train' if self.train else 'test'
        if mode == 'train':
            dataset = np.load(os.path.join(self.root,
                                           'wav{}.npz'.format(self.sr // 1000)),
                              allow_pickle=True);
            train_sounds = []
            train_labels = []
            for i in range(1, self.nFolds + 1):
                sounds = dataset['fold{}'.format(i)].item()['sounds']
                labels = dataset['fold{}'.format(i)].item()['labels']
                if i != self.fold:
                    train_sounds.extend(sounds)
                    train_labels.extend(labels)

            self.data = []
            self.targets = []  # labels of 1-50

            for i in range(len(train_sounds)):
                sound, target = train_sounds[i], train_labels[i]
                sound = self.preprocess(sound).astype(np.float32)
                label = np.zeros(self.nClasses)
                label[target] = 1

                self.data.append(sound)
                self.targets.append(target)

            self.data = np.asarray(self.data)
            self.data = np.expand_dims(self.data, axis=1)
            self.data = np.expand_dims(self.data, axis=3)
            self.targets = np.asarray(self.targets, dtype=int)

            print('training dataset')
            print(self.data.shape)  # Should be (1600, 1, 30225, 1)
            print(self.targets.shape)  # Should be (1600,)

        else:
            for fold in range(1, 6):
                data = np.load(os.path.join(self.root,
                                            'test_data_{}khz/fold{}_test4000.npz'.format(
                                                self.sr // 1000, fold)),
                               allow_pickle=True)

                if fold == 1:
                    self.data = data['x']
                    self.targets = data['y']
                else:
                    self.data = np.concatenate((self.data, data['x']), axis=0)
                    self.targets = np.concatenate((self.targets, data['y']), axis=0)

            print('testing dataset')
            print(self.data.shape)  # Should be (400, 1, 30225, 1)
            print(self.targets.shape)  # Should be (400,)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def preprocess_setup(self):
        funcs = []

        funcs += [U.padding(self.inputLength // 2),
                  U.random_crop(self.inputLength),
                  U.normalize(32768.0)]
        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound