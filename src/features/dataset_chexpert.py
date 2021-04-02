import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class CheXPert(Dataset):

    def __init__(self, path_to_data, fold, uncertainty=False, transform=None, orientation="all"):
        """ Dataset class for CheXPert

        Args:
            path_to_data (string): Path to dataset
            fold (string): train, valid, test, all
            uncertainty (bool): Changes label calculation
            transform (torchvision.Transforms): transforms performed on the samples
            orientation (string): "all", "frontal", "lateral"
        """

        self.uncertainty = uncertainty
        self.transform = transform
        self.path_to_data = path_to_data
        self.orientation = orientation

        self.PRED_LABEL = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices"]

        # Loading the correct CSV for train, valid, test or all
        if fold == "train":
            self.labels = pd.read_csv(os.path.join(path_to_data, "train_fold.csv"))
        elif fold == "valid":
            self.labels = pd.read_csv(os.path.join(path_to_data, "valid_fold.csv"))
        elif fold == "test":
            self.labels = pd.read_csv(os.path.join(path_to_data, "test.csv"))
        elif fold == "all":
            self.labels = pd.read_csv(os.path.join(path_to_data, "train_fold.csv"))
            tmp1 = pd.read_csv(os.path.join(path_to_data, "valid_fold.csv"))
            tmp2 = pd.read_csv(os.path.join(path_to_data, "test.csv"))
            self.labels = self.labels.append(tmp1, ignore_index=True)
            self.labels = self.labels.append(tmp2, ignore_index=True)
        else:
            raise Exception("Wrong fold input given!")

        # Deleting either lateral or frontal images of the Dataset or keep all
        if self.orientation == "lateral":
            self.labels = self.labels[~self.labels.Path.str.contains("frontal")]
        elif self.orientation == "frontal":
            self.labels = self.labels[~self.labels.Path.str.contains("lateral")]
        elif self.orientation == "all":
            pass
        else:
            raise Exception("Wrong orientation input given!")

    def get_weights(self):
        """
        Returns class weights for weighted loss

        Returns:
            (torch.Tensor): Array of size 14 with class weigths
         """

        weights = np.zeros(len(self.PRED_LABEL))
        all = len(self.labels)

        for i, label in enumerate(self.PRED_LABEL):
            values = np.asarray(self.labels[label])
            positive = np.count_nonzero(values == 1.0)
            weights[i] = (all - positive) / positive

        return torch.from_numpy(weights.astype(np.float32))

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

        image_path = os.path.join(self.path_to_data, self.labels.iloc[idx]['Path'])

        image = Image.open(image_path).convert('RGB')

        # Get labels from the dataframe for current image
        label = self.labels.iloc[idx, :].loc[self.PRED_LABEL]
        label = label.to_numpy()

        # Uncertainty labels are mapped to 0.0
        if not self.uncertainty:
            tmp = np.zeros(len(self.PRED_LABEL))
            tmp[label == 1] = 1.0
            label = tmp

        # Calls the transforms on image
        if self.transform:
            image = self.transform(image)

        return image, label

