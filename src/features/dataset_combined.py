import os
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from src.features.dataset_nih import NIH_full
from src.features.dataset_chexpert import CheXPert
from src.features.dataset_brixia import BrixIA
from src.__init__ import ROOT_DIR


class CombinedData(ConcatDataset):

    def __init__(self, path_to_nih, path_to_brixia, path_to_chexpert, fold, transform=None, orientation="frontal"):
        self.nih = NIH_full(path_to_nih, fold, transform=transform)
        self.brixia = BrixIA(path_to_brixia, fold, 'paper', transform=transform)
        self.chexpert = CheXPert(path_to_chexpert, fold, transform=transform, orientation=orientation)

        self.PRED_LABEL = ['No Finding', 'Pathology']

        super().__init__([self.nih, self.chexpert, self.brixia])

    def __getitem__(self, idx):
        """
        Arguments:
        - idx (int) : Index of the image to return
        Returns:
        - image (PIL.Image): PIL format image
        """
        sample = super().__getitem__(idx)
        img = sample[0]
        label = sample[1]
        label = self._get_label(idx, label)

        return img, float(label)

    def _get_label(self, idx, label):
        """ Used to convert the original labels of the datasets to 'No Finding':0 and 'Pathology':1

        Arguments:
        - idx (int) : Index of the image to return
        - idx (label) : The label from the original dataset
        Returns:
        - (int): 0 or 1
        """
        if 0 <= idx < len(self.datasets[0]):
            return 0 if torch.sum(label) == 0 else 1
        elif len(self.datasets[0]) <= idx < len(self.datasets[0]) + len(self.datasets[1]):
            return 0 if label[0] == 1 else 1
        else:
            label = np.argmax(label, axis=1)
            label = np.sum(label, axis=0)
            return 0 if label == 0 else 1

