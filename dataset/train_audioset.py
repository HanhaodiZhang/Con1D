import h5py
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from einops import rearrange


def get_h5_file_list(h5_dictionary, train=True):
    num_files = len(os.listdir(h5_dictionary))
    h5_list = []
    for i in range(num_files):
        if train:
            h5_list.append(os.path.join(h5_dictionary, f"train_balanced_{i + 1}.h5"))
        else:
            h5_list.append(os.path.join(h5_dictionary, f"eval_{i + 1}.h5"))
    return h5_list


class AudioDataset(Dataset):
    def __init__(self, h5_dictionary,train =True):
        h5file_path_list = get_h5_file_list(h5_dictionary, train)
        self.h5_list = [h5py.File(h5file_path, 'r') for h5file_path in h5file_path_list]
        self.h5_path_list = h5file_path_list
        self.len = sum(len(h5['wav_data']) for h5 in self.h5_list)
        print(self.len)

    def __getitem__(self, idx):
        wav = torch.from_numpy(self.h5_list[idx // 5000]['wav_data'][idx % 5000])
        label = torch.from_numpy(self.h5_list[idx // 5000]['label_data'][idx % 5000])
        return wav, label

    def __len__(self):
        return self.len
