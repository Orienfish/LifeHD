""" Core50 Pytorch Dataset """
""" Adapted from ContinualAI https://avalanche.continualai.org/"""

import glob
import logging
import numpy as np
import os
import pickle as pkl
from pathlib import Path
from typing import Optional, Union

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, extract_archive

from . import core50_data

# Convert ordering (in stream51) to scenario (in core50)
order2scen = {
    "instance": "ni",
    "class_iid": "nc",
    "class_instance": "nic" 
}

nbatch = {
    "ni": 8,
    "nc": 9,
    "nic": 79,
    "nicv2_79": 79,
    "nicv2_196": 196,
    "nicv2_391": 391,
}

scen2dirs = {
    "ni": "batches_filelists/NI_inc/",
    "nc": "batches_filelists/NC_inc/",
    "nic": "batches_filelists/NIC_inc/",
    "nicv2_79": "NIC_v2_79/",
    "nicv2_196": "NIC_v2_196/",
    "nicv2_391": "NIC_v2_391/",
}

class CORe50(Dataset):
    """CORe50 Pytorch Dataset"""

    def __init__(
        self,
        root=None,
        ordering='class_instance',
        train=True,
        transform=None,
        target_transform=None,
        sample_ratio=1.0,
        loader=default_loader,
        load_at_beginning=False,
        download=True,
        mini=False,
        object_level=True,
    ):

        """Creates an instance of the CORe50 dataset.

        :param root: root for the datasets data. Defaults to None, which
            means that the default location for 'core50' will be used.
        :param ordering: The dataset ordering must be one of: "iid", "class_iid", 
        "instance", or "class_instance"
        :param train: train or test split.
        :param transform: eventual transformations to be applied.
        :param target_transform: eventual transformation to be applied to the
            targets.
        :param sample_ratio: The ratio of the subset to sample
        :param loader: the procedure to load the instance from the storage.
        :param load_at_beginning: If True, the sampled subset will be loaded
            at the beginning during __init__. If False, it will be loaded
            during __getitem__.
        :param download: boolean to automatically download data. Default to
            True.
        :param mini: boolean to use the 32x32 version instead of the 128x128.
            Default to False.
        :param object_level: if the classification is objects based or
            category based: 50 or 10 way classification problem. Default to True
            (50-way object classification problem)
        """
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.sample_ratio = sample_ratio
        self.loader = loader
        self.load_at_beginning = load_at_beginning
        self.download = download
        self.object_level = object_level
        self.mini = mini
        self.verbose = True

        # any scenario and run is good here since we want just to load the
        # train images and targets with no particular order
        # :param scenario: CORe50 main scenario. It can be chosen between 'ni', 'nc',
        # 'nic', 'nicv2_79', 'nicv2_196' or 'nicv2_391.'
        # :param run: number of run for the benchmark. Each run defines a different
        # ordering. Must be a number between 0 and 9.
        self._scen = order2scen[ordering]
        self._run = 0
        self._nbatch = nbatch[order2scen[ordering]]

        # Download the dataset and initialize metadata
        self._load_dataset()

        # Sample the subset and load all samples
        if self.load_at_beginning:
            self.sampled_index_list = np.random.choice(len(self.targets),
                                                       size=int(self.sample_ratio * len(self.targets)),
                                                       replace=False)
            self.data, self.new_targets = [], []
            if self.mini:
                bp = "core50_32x32"
            else:
                bp = "core50_128x128"

            for index in self.sampled_index_list:
                print(index, '/', len(self.sampled_index_list))
                sample = self.loader(os.path.join(self.root, bp + '/' +
                                                  self.paths[index]))
                self.data.append(sample)
                self.new_targets.append(self.targets[index])

    def _load_dataset(self) -> None:
        """
        The standardized dataset download and load procedure.

        For more details on the coded procedure see the class documentation.

        This method shouldn't be overridden.

        This method will raise and error if the dataset couldn't be loaded
        or downloaded.

        :return: None
        """
        metadata_loaded = False
        metadata_load_error = None
        try:
            metadata_loaded = self._load_metadata()
        except Exception as e:
            metadata_load_error = e

        if metadata_loaded:
            if self.verbose:
                print("Files already downloaded and verified")
            return

        if not self.download:
            msg = (
                "Error loading dataset metadata (dataset download was "
                'not attempted as "download" is set to False)'
            )
            if metadata_load_error is None:
                raise RuntimeError(msg)
            else:
                print(msg)
                raise metadata_load_error

        try:
            self._download_dataset()
        except Exception as e:
            err_msg = self._download_error_message()
            print(err_msg, flush=True)
            raise e

        if not self._load_metadata():
            err_msg = self._download_error_message()
            print(err_msg)
            raise RuntimeError(
                "Error loading dataset metadata (... but the download "
                "procedure completed successfully)"
            )

    def _download_file(
        self, url: str, file_name: str, checksum: Optional[str]
    ) -> Path:
        """
        Utility method that can be used to download and verify a file.

        :param url: The download url.
        :param file_name: The name of the file to save. The file will be saved
            in the `root` with this name. Always fill this parameter.
            Don't pass a path! Pass a file name only!
        :param checksum: The MD5 hash to use when verifying the downloaded
            file. Can be None, in which case the check will be skipped.
            It is recommended to always fill this parameter.
        :return: The path to the downloaded file.
        """
        if os.path.exists(os.path.join(self.root, file_name)):
            print('Download {} not needed, files already on disk.'.format(file_name))
        else:
            download_url(url, str(self.root), filename=file_name, md5=checksum)
        return os.path.join(self.root, file_name)

    def _extract_archive(
        self,
        path: Union[str, Path],
        sub_directory: str = None,
        remove_archive: bool = False,
    ) -> Path:
        """
        Utility method that can be used to extract an archive.

        :param path: The complete path to the archive (for instance obtained
            by calling `_download_file`).
        :param sub_directory: The name of the sub directory where to extract the
            archive. Can be None, which means that the archive will be extracted
            in the root. Beware that some archives already have a root directory
            inside of them, in which case it's probably better to use None here.
            Defaults to None.
        :param remove_archive: If True, the archive will be deleted after a
            successful extraction. Defaults to False.
        :return:
        """

        if sub_directory is None:
            extract_root = self.root
        else:
            extract_root = os.path.join(self.root, sub_directory)

        extract_archive(
            str(path), to_path=str(extract_root), remove_finished=remove_archive
        )

        return extract_root

    def _download_dataset(self) -> None:
        data2download = core50_data.data

        if self.mini:
            data2download = list(data2download)
            data2download[0] = core50_data.extra_data[1]

        for name in data2download:
            if self.verbose:
                print("Downloading " + name[1] + "...")
            file = self._download_file(name[1], name[0], name[2])
            if name[1].endswith(".zip"):
                if self.verbose:
                    print(f"Extracting {name[0]}...")
                extract_root = self._extract_archive(file)
                if self.verbose:
                    print("Extraction completed!")

    def _load_metadata(self) -> bool:
        if self.mini:
            bp = "core50_32x32"
        else:
            bp = "core50_128x128"

        if not os.path.exists(os.path.join(self.root, bp)):
            return False

        if not os.path.exists(os.path.join(self.root, "batches_filelists")):
            return False

        with open(os.path.join(self.root, "paths.pkl"), "rb") as f:
            self.train_test_paths = pkl.load(f)

        if self.verbose:
            print("Loading labels...")
        with open(os.path.join(self.root, "labels.pkl"), "rb") as f:
            self.all_targets = pkl.load(f)
            self.train_test_targets = []
            for i in range(self._nbatch + 1):
                self.train_test_targets += self.all_targets[self._scen][
                    self._run
                ][i]

        if self.verbose:
            print("Loading LUP...")
        with open(os.path.join(self.root, "LUP.pkl"), "rb") as f:
            self.LUP = pkl.load(f)

        if self.verbose:
            print("Loading labels names...")
        with open(os.path.join(self.root, "labels2names.pkl"), "rb") as f:
            self.labels2names = pkl.load(f)

        self.idx_list = []
        if self.train:
            for i in range(self._nbatch):
                self.idx_list += self.LUP[self._scen][self._run][i]
        else:
            self.idx_list = self.LUP[self._scen][self._run][-1]

        self.paths = []
        self.targets = []

        for idx in self.idx_list:
            self.paths.append(self.train_test_paths[idx])
            div = 1
            if not self.object_level:
                div = 5
            self.targets.append(self.train_test_targets[idx] // div)

        with open(os.path.join(self.root, "labels2names.pkl"), "rb") as f:
            self.labels2names = pkl.load(f)

        if not os.path.exists(os.path.join(self.root, "NIC_v2_79_cat")):
            self._create_cat_filelists()

        return True

    def _download_error_message(self) -> str:
        all_urls = [name_url[1] for name_url in core50_data.data]

        base_msg = (
            "[CORe50] Error downloading the dataset!\n"
            "You should download data manually using the following links:\n"
        )

        for url in all_urls:
            base_msg += url
            base_msg += "\n"

        base_msg += "and place these files in " + str(self.root)

        return base_msg

    def _create_cat_filelists(self):
        """Generates corresponding filelists with category-wise labels. The
        default one are based on the object-level labels from 0 to 49."""

        for k, v in core50_data.scen2dirs.items():
            orig_root_path = os.path.join(self.root, v)
            root_path = os.path.join(self.root, v[:-1] + "_cat")
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            for run in range(10):
                cur_path = os.path.join(root_path, "run" + str(run))
                orig_cur_path = os.path.join(orig_root_path, "run" + str(run))
                if not os.path.exists(cur_path):
                    os.makedirs(cur_path)
                for file in glob.glob(os.path.join(orig_cur_path, "*.txt")):
                    o_filename = file
                    _, d_filename = os.path.split(o_filename)
                    orig_f = open(o_filename, "r")
                    dst_f = open(os.path.join(cur_path, d_filename), "w")
                    for line in orig_f:
                        path, label = line.split(" ")
                        new_label = self._objlab2cat(int(label), k, run)
                        dst_f.write(path + " " + str(new_label) + "\n")
                    orig_f.close()
                    dst_f.close()

    def _objlab2cat(self, label, scen, run):
        """Mapping an object label into its corresponding category label
        based on the scenario."""

        if scen == "nc":
            return core50_data.name2cat[
                self.labels2names["nc"][run][label][:-1]
            ]
        else:
            return int(label) // 5

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target
                class.
        """
        if self.load_at_beginning:
            img, target = self.data[index], self.new_targets[index]
        else:
            target = self.targets[index]
            if self.mini:
                bp = "core50_32x32"
            else:
                bp = "core50_128x128"

            img = self.loader(os.path.join(self.root, bp + '/' + self.paths[index]))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.load_at_beginning:
            return len(self.new_targets)
        else:
            return len(self.targets)