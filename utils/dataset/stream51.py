""" Stream-51 Pytorch Dataset """
""" Adapted from ContinualAI https://avalanche.continualai.org/"""

import os
import numpy as np
import shutil
import json
import random
from pathlib import Path
from typing import Optional

from torchvision.datasets.folder import default_loader
from zipfile import ZipFile
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

stream51_src = (
    "Stream-51.zip",
    "http://klab.cis.rit.edu/files/Stream-51.zip",
    "5f34d542b71af7e5ecb2226d1bc7297c",
)


class Stream51(Dataset):
    """Stream-51 Pytorch Dataset"""

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
        download=True
    ):
        """
        Creates an instance of the Stream-51 dataset.
        :param root: The directory where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            'stream51' will be used.
        :param ordering: The dataset ordering must be one of: "iid", "class_iid", 
            "instance", or "class_instance"
        :param train: If True, the training set will be returned. If False,
            the test set will be returned.
        :param transform: The transformations to apply to the X values.
        :param target_transform: The transformations to apply to the Y values.
        :param sample_ratio: The ratio of the subset to sample
        :param loader: The image loader to use.
        :param load_at_beginning: If True, the sampled subset will be loaded
            at the beginning during __init__. If False, it will be loaded
            during __getitem__.
        :param download: If True, the dataset will be downloaded if needed.
        """
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.sample_ratio = sample_ratio
        self.loader = loader
        self.load_at_beginning = load_at_beginning
        self.download = download
        self.bbox_crop = True
        self.verbose = True
        self.ratio = 1.1

        self._load_dataset()

        # Call make_dataset to organize the order in training dataset
        if train:
            self.make_dataset(ordering)

        # Sample the subset and load all samples
        if self.load_at_beginning:
            self.sampled_index_list = np.random.choice(len(self.samples),
                                                       size=int(self.sample_ratio * len(self.samples)),
                                                       replace=False)
            self.data, self.new_targets = [], []
            for index in self.sampled_index_list:
                print(index, '/', len(self.sampled_index_list))
                fpath = self.samples[index][-1]
                sample = self.loader(os.path.join(self.root, fpath))

                if self.bbox_crop:
                    bbox = self.samples[index][-2]
                    cw = bbox[0] - bbox[1]
                    ch = bbox[2] - bbox[3]
                    center = [int(bbox[1] + cw / 2), int(bbox[3] + ch / 2)]
                    bbox = [
                        min([int(center[0] + (cw * self.ratio / 2)),
                             sample.size[0]]),
                        max([int(center[0] - (cw * self.ratio / 2)), 0]),
                        min([int(center[1] + (ch * self.ratio / 2)),
                             sample.size[1]]),
                        max([int(center[1] - (ch * self.ratio / 2)), 0]),
                    ]
                    sample = sample.crop((bbox[1], bbox[3], bbox[0], bbox[2]))

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

    def _download_dataset(self) -> None:
        self._download_file(
            stream51_src[1], stream51_src[0], stream51_src[2]
        )

        if self.verbose:
            print("[Stream-51] Extracting dataset...")

        if stream51_src[1].endswith(".zip"):
            lfilename = os.path.join(self.root, stream51_src[0])
            with ZipFile(str(lfilename), "r") as zipf:
                for member in zipf.namelist():
                    filename = os.path.basename(member)
                    # skip directories
                    if not filename:
                        continue

                    # copy file (taken from zipfile's extract)
                    source = zipf.open(member)
                    if "json" in filename:
                        target = open(os.path.join(self.root, filename), "wb")
                    else:
                        dest_folder = os.path.join(
                            *(member.split(os.path.sep)[1:-1])
                        )
                        dest_folder = os.path.join(self.root, dest_folder)
                        os.makedirs(dest_folder, exist_ok=True)

                        target = open(os.path.join(dest_folder, filename), "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)

            # lfilename.unlink()

    def _load_metadata(self) -> bool:
        if self.train:
            data_list = json.load(
                open(os.path.join(self.root, "Stream-51_meta_train.json"))
            )
        else:
            data_list = json.load(
                open(os.path.join(self.root, "Stream-51_meta_test.json"))
            )
            # Filter out novelty detection in the test dataset
            ind = [i for i in range(len(data_list)) if data_list[i][0] < 51]
            data_list = [data_list[i] for i in ind]

        self.samples = data_list
        self.targets = [s[0] for s in data_list]

        self.bbox_crop = True
        self.ratio = 1.1

        return True

    def _download_error_message(self) -> str:
        return (
            "[Stream-51] Error downloading the dataset. Consider "
            "downloading it manually at: "
            + stream51_src[1]
            + " and placing it in: "
            + str(self.root)
        )

    @staticmethod
    def _instance_ordering(data_list):
        # organize data by video
        total_videos = 0
        new_data_list = []
        temp_video = []
        for x in data_list:
            if x[3] == 0:
                new_data_list.append(temp_video)
                total_videos += 1
                temp_video = [x]
            else:
                temp_video.append(x)
        new_data_list.append(temp_video)
        new_data_list = new_data_list[1:]
        # shuffle videos
        random.shuffle(new_data_list)
        # reorganize by clip
        data_list = []
        for v in new_data_list:
            for x in v:
                data_list.append(x)
        return data_list

    @staticmethod
    def _class_ordering(data_list, class_type):
        # organize data by class
        new_data_list = []
        for class_id in range(data_list[-1][0] + 1):
            class_data_list = [x for x in data_list if x[0] == class_id]
            if class_type == "class_iid":
                # shuffle all class data
                random.shuffle(class_data_list)
            else:
                # shuffle clips within class
                class_data_list = Stream51._instance_ordering(
                    class_data_list
                )
            new_data_list.append(class_data_list)
        # shuffle classes
        random.shuffle(new_data_list)
        # reorganize by class
        data_list = []
        for v in new_data_list:
            for x in v:
                data_list.append(x)
        return data_list

    def make_dataset(self, ordering="class_instance"):
        """
        data_list
        for train: [class_id, clip_num, video_num, frame_num, bbox, file_loc]
        for test: [class_id, bbox, file_loc]
        """
        data_list = self.samples
        if not ordering or len(data_list[0]) == 3:  # cannot order the test set
            return
        if ordering not in ["iid", "class_iid", "instance", "class_instance"]:
            raise ValueError(
                'dataset ordering must be one of: "iid", "class_iid", '
                '"instance", or "class_instance"'
            )
        if ordering == "iid":
            # shuffle all data
            random.shuffle(data_list)
        elif ordering == "instance":
            data_list = self._instance_ordering(data_list)
        elif "class" in ordering:
            data_list = self._class_ordering(data_list, ordering)

        self.samples = data_list
        self.targets = [s[0] for s in data_list]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target
            class.
        """
        if self.load_at_beginning:
            sample, target = self.data[index], self.new_targets[index]
        else:
            fpath, target = self.samples[index][-1], self.targets[index]
            sample = self.loader(os.path.join(self.root, fpath))

            if self.bbox_crop:
                bbox = self.samples[index][-2]
                cw = bbox[0] - bbox[1]
                ch = bbox[2] - bbox[3]
                center = [int(bbox[1] + cw / 2), int(bbox[3] + ch / 2)]
                bbox = [
                    min([int(center[0] + (cw * self.ratio / 2)),
                         sample.size[0]]),
                    max([int(center[0] - (cw * self.ratio / 2)), 0]),
                    min([int(center[1] + (ch * self.ratio / 2)),
                         sample.size[1]]),
                    max([int(center[1] - (ch * self.ratio / 2)), 0]),
                ]
                sample = sample.crop((bbox[1], bbox[3], bbox[0], bbox[2]))

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.load_at_beginning:
            return len(self.new_targets)
        else:
            return len(self.targets)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )

        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp,
            self.target_transform.__repr__().replace(
                "\n", "\n" + " " * len(tmp)
            ),
        )
        return fmt_str