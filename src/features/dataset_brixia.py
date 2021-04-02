import pandas as pd
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BrixIA(Dataset):

    def __init__(self, path_to_data, fold, mode, transform=None):
        """ Dataset class for BrixIA

        Args:
            path_to_data (string): Path to dataset
            fold (string): train, valid, test, all
            transform (torchvision.Transforms): transforms performed on the samples
            mode (string): 'paper': 6 zone regression
                           'regression': single target regression
        """

        self.transform = transform
        self.path_to_data = path_to_data
        self.fold = fold
        self.mode = mode

        self.labels = pd.read_csv(os.path.join(path_to_data, 'metadata_global_v2.csv'), sep=';',
                                  dtype={'BrixiaScore': str})

        self.mean = np.mean(self.labels['BrixiaScoreGlobal'])
        self.std = np.mean(self.labels['BrixiaScoreGlobal'])

        if self.mode == 'paper':
            self.PRED_LABEL = [
                'BrixiaScore',
            ]
        elif self.mode == 'regression':
            self.PRED_LABEL = [
                'BrixiaScoreGlobal',
            ]

            # Weights for weighted MSE
            weights = np.empty(19)
            weights.fill(len(self.labels['BrixiaScoreGlobal']))
            class_count = self.labels['BrixiaScoreGlobal'].value_counts().sort_index()
            weights /= class_count
            weights /= weights.max()
            self.weights = weights

        if self.fold == 'train':
            self.labels = self.labels[self.labels['ConsensusTestset'] == 0].sample(frac=1)[:int(len(self.labels) * 0.9)]
        if self.fold == 'valid':
            self.labels = self.labels[self.labels['ConsensusTestset'] == 0].sample(frac=1)[int(len(self.labels) * 0.9):]
        if self.fold == 'test':
            self.labels = self.labels[self.labels['ConsensusTestset'] == 1]

    def __len__(self):
        """
        Returns:
            (int): length of the pandas dataframe
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Arguments:
        - idx (int) : Index of the image to return
        Returns:
        - image (PIL.Image): PIL format image
        """

        image_path = os.path.join(self.path_to_data, 'images', self.labels.iloc[idx]['Filename']).replace('.dcm',
                                                                                                          '.jpg')

        image = Image.open(image_path).convert('RGB')

        # Get labels from the dataframe for current image
        label = self.labels.iloc[idx, :].loc[self.PRED_LABEL]
        weight = 1

        if self.mode == 'paper':
            label = np.array(list(label.to_numpy()[0]), dtype=int)
            label_tmp = np.zeros((6, 4))
            label_tmp[np.arange(label.size), label] = 1
            label = label_tmp

        elif self.mode == 'regression':
            weight = self.weights[int(label)]
            label = torch.tensor((label - self.mean) / self.std).type(torch.float32)

        # Calls the transforms on image
        if self.transform:
            image = self.transform(image)

        return image, label, weight
