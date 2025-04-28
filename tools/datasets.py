import torch
import numpy as np
from torch.utils.data import Dataset
import random

class EEGset(Dataset):
    def __init__(self, root_path, pick_id=(1,), settup='train', T=125, EA=False, loo=False, all_id=range(1, 10)):
        self.T = T
        data_path = root_path + settup + '/'
        if loo:
            if EA:
                data_path = data_path + 'EA/'
            if settup == 'test':
                self.data_info = self.load(data_path, pick_id)
            else:
                train_id = list(set(all_id) - set(pick_id))
                self.data_info = self.load(data_path, train_id)
        else:
            self.data_info = self.load(data_path, pick_id)

    def __getitem__(self, index):
        img, label = self.data_info[index]
        return img, label

    def __len__(self):
        return len(self.data_info)

    def Integrate(self, data, start_t=0):  # [C, T]
        return data[:, start_t:start_t+self.T].astype(np.float32)

    def load(self, data_path, pick_id):
        data_info = list()
        for person_id in pick_id:
            data = np.load(data_path + f'data_id{person_id}.npy', allow_pickle=True)
            labels = np.load(data_path + f'label_id{person_id}.npy', allow_pickle=True)
            num = data.shape[0]
            T = data.shape[-1]
            if T == self.T:
                data_info += list((data[i].astype(np.float32), labels[i]) for i in range(num))
            else:
                for i in range(num):
                    EEG = data[i]
                    label = labels[i]
                    data_info.append((self.Integrate(EEG), label))
        return data_info