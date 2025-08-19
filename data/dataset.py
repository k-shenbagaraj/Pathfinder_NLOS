import os
import random

import natsort
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split

from .loader import npy_loader


class NLOSNet_Dataloader(Dataset):
    """
    Works with multiple patches
    """

    def __init__(
        self, dataset_root: str, route_len: int = 8, shuffle: bool = True, **kwargs
    ) -> None:
        self.dataset_root = dataset_root
        self.dirs = os.listdir(self.dataset_root)
        self.total_len = len(self.dirs)
        self.num_frames = route_len
        if shuffle:
            random.shuffle(self.dirs)

    def __len__(self):
        print("len(self.dirs): ", len(self.dirs))
        return len(self.dirs)

    def __getitem__(self, idx):
        p_dir = self.dataset_root + "/" + self.dirs[idx]
        len_dir = len(os.listdir(p_dir))
        num_planes = 3
        routes = np.zeros((len_dir, num_planes, 2))
        v_gt = np.zeros((len_dir, num_planes, 1))
        map_sizes = np.zeros((len_dir, 1, 2))
        video = []
        cnt = 0
        planeid = []
        diffimg = []

        pos0 = np.zeros((1, 2))
        for i in range(len_dir):
            mat_file = loadmat(os.path.join(p_dir, str(i) + ".mat"))  
            raw = torch.Tensor(mat_file["raw"])
            diff = torch.Tensor(mat_file["diff"])
            route = mat_file["nlos"].astype("float32")
            timediff = route[0][0]
            route = route[0][1:3]
            pos1 = route
            # Calculate distance between two points
            dist = np.linalg.norm(pos1 - pos0)
            pos0 = pos1
            route = np.expand_dims(route, axis=0)
            map_size = np.array([[max(route[:, 0]), max(route[:, 1])]])

            planeid.append(torch.Tensor(mat_file["plane_ids"]))
            routes[cnt, 0, :] = route
            routes[cnt, 1, :] = route
            routes[cnt, 2, :] = route
            # For the first frame, the velocity is zero
            if cnt == 0:
                v_gt[cnt, 0, :] = 0
                v_gt[cnt, 1, :] = 0
                v_gt[cnt, 2, :] = 0
            else:
                v_ = dist / timediff
                v_gt[cnt, 0, :] = v_
                v_gt[cnt, 1, :] = v_
                v_gt[cnt, 2, :] = v_
            map_sizes[cnt, 0, :] = map_size
            cnt += 1
            video.append(raw)
            diffimg.append(diff)
        map_sizes = map_sizes.astype("float32")
        return (
            video,
            diffimg,
            planeid,
            torch.Tensor(routes).float(),
            torch.Tensor(v_gt).float(),
            torch.Tensor(map_sizes).float(),
        )


def split_dataset(phase: str = "train", train_ratio: float = 0.9, **kwargs):
    full_dataset = NLOSNet_Dataloader(**kwargs)
    if phase == "train":
        train_size = int(len(full_dataset) * train_ratio)
        val_size = len(full_dataset) - train_size
        return random_split(full_dataset, [train_size, val_size])
    elif phase == "test":
        return full_dataset
