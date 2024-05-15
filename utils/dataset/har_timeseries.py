import os
import numpy as np
import math
from torchvision import transforms
from torch.utils.data import Dataset
import scipy.io

def sliding_window(data, targets, win_size, overlap):
    processed_data = []
    processed_targets = []
    step = math.floor(win_size - overlap * win_size)
    for t in range(win_size, len(data), step):
        processed_data.append(data[t-win_size:t])
        processed_targets.append(np.bincount(targets[t-win_size:t]).argmax())
    processed_data = np.stack(processed_data)
    processed_targets = np.asarray(processed_targets, dtype=np.int32)
    return processed_data, processed_targets

class HAR_TimeSeries(Dataset):
    """
    Defines HAR time series dataset as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False,
                 test_partition: float=0.2, win_size: int=128, overlap: float=0.75) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.test_partition = test_partition
        self.win_size = win_size
        self.overlap = overlap

        mode = 'train' if self.train else 'test'

        """
        # Another setup option: directly download from uci repo
        # But the accuracy is lower with 23 sensors instead of 21 in mhealth.mat
        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                if not os.path.exists(root):
                    os.mkdir(root)
                cmd = 'wget -O {} {}'.format(
                    os.path.join(root, 'MHEALTH.zip'),
                    'http://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip'
                )
                os.system(cmd)
                os.system('unzip -o {} -d {}'.format(
                    os.path.join(root, 'MHEALTH.zip'),
                    root
                ))

        data = []
        targets = []

        for id in range(1, 10):
            new_data = np.loadtxt(
                os.path.join(root, 'MHEALTHDATASET/mHealth_subject%d.log' % (id)),
                dtype='float32'
            )
            data.append(new_data[:, :-1])
            targets.append(new_data[:, -1])
            
        data = np.concatenate(data, axis=0)
        targets = np.concatenate(targets, axis=0).astype('int')
        """

        mat = scipy.io.loadmat(os.path.join(root, 'har.mat'))

        acce = np.concatenate([mat['acce_train'],mat['acce_test']], axis=0)
        gyro = np.concatenate([mat['gyro_train'],mat['gyro_test']], axis=0)
        all_data = np.concatenate([acce, gyro], axis=1).astype(np.float32)
        all_y = np.concatenate([mat['y_train'].squeeze(),mat['y_test'].squeeze()], 
                               axis=0).astype(np.int32) - 1

        # Normalize
        all_data = (all_data - np.min(all_data, axis=0)) / \
            (np.max(all_data, axis=0) - np.min(all_data, axis=0))
        
        #print('before sliding window: ', all_data.shape, all_y.shape)
        self.data, self.targets = sliding_window(all_data, all_y, 
                                                 self.win_size, 
                                                 self.overlap)
        
        # Partition training and testing samples randomly, and save the testing indexes
        test_samples_num = int(self.data.shape[0] * self.test_partition)
        test_samples_idx = np.random.choice(np.arange(self.data.shape[0]),
                                           size=test_samples_num,
                                           replace=False).astype('int')
        if mode == 'train':
            self.data = np.delete(self.data, test_samples_idx, axis=0)
            self.targets = np.delete(self.targets, test_samples_idx, axis=0)
            np.savetxt(os.path.join(root, 'test_samples_idx.txt'),
                   test_samples_idx)
            print(self.targets)
        else:  # test
            test_samples_idx = np.loadtxt(os.path.join(root, 'test_samples_idx.txt')).astype('int')
            self.data = self.data[test_samples_idx]
            self.targets = self.targets[test_samples_idx]

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
    