import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

class ISOLET(Dataset):
    """
    Defines ISOLET pytorch datasets.
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
                cmd = 'wget -O {} {}; uncompress {};'.format(
                    os.path.join(root, 'isolet_train.data.Z'),
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet1+2+3+4.data.Z',
                    os.path.join(root, 'isolet_train.data.Z')
                )
                os.system(cmd)
                cmd = 'wget -O {} {}; uncompress {};'.format(
                    os.path.join(root, 'isolet_test.data.Z'),
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet5.data.Z',
                    os.path.join(root, 'isolet_test.data.Z')
                )
                os.system(cmd)

        mode = 'train' if self.train else 'test'
        data = np.loadtxt(
            os.path.join(root, 'isolet_%s.data' % mode),
            delimiter=',',
            dtype='float32'
        )
        self.data = data[:, :-1]
        self.targets = data[:, -1].astype('long')
        self.targets = self.targets - 1  # Move class label 1-26 to 0-25

        # Save the mean and std of HAR dataset and normalize the raw features
        # Normalize data to [0,1]
        self.data = self.data = (self.data - np.min(self.data, axis=0)) / \
            (np.max(self.data, axis=0) - np.min(self.data, axis=0))
        self.data = np.expand_dims(self.data, axis=2)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        if self.transform is not None:
            data= self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target