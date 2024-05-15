import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

class HAR(Dataset):
    """
    Defines Human Activity Recognition as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                if not os.path.exists(root):
                    os.mkdir(root)
                cmd = 'wget -O {} {}'.format(
                    os.path.join(root, 'HAR.zip'),
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
                )
                os.system(cmd)
                os.system('unzip -o {} -d {}'.format(
                    os.path.join(root, 'HAR.zip'),
                    root
                ))

        mode = 'train' if self.train else 'test'
        self.data = np.loadtxt(
            os.path.join(root, 'UCI HAR Dataset/%s/X_%s.txt' % (mode, mode)),
            dtype=np.float32
        )
        self.targets = np.loadtxt(
            os.path.join(root, 'UCI HAR Dataset/%s/y_%s.txt' % (mode, mode)),
            dtype=np.int32
        )
        self.targets = self.targets - 1  # Move class label 1-6 to 0-5

        # Normalize data to [0,1]
        self.data = (self.data - np.min(self.data, axis=0)) / \
            (np.max(self.data, axis=0) - np.min(self.data, axis=0))
        # Expand dimension for match the size of images in to_tensor transform
        self.data = np.expand_dims(self.data, axis=2)

        print('max: ', np.max(self.data), 'min: ', np.min(self.data))
        print(mode, self.data.shape, self.targets.shape)
        print(self.data.dtype, self.targets.dtype)
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target