import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as torch_transforms
import math

class CovidSeg(Dataset):

    def __init__(self, path_to_data, transform=None, output_label_size = None):
        """ Dataset class for covid segmentation dataset

        Args:
            path_to_data (string): Path to dataset
            transform (torchvision.Transforms): transforms performed on the samples
        """

        self.transform = transform
        self.path_to_data = path_to_data
        self.output_label_size = output_label_size

        self.PRED_LABEL = {'Left Lung': 0,
                           'Right Lung': 1,
                           'Cardiomediastinum': 2,
                           'Airways': 3,
                           'Ground Glass Opacities': 4,
                           'Consolidation': 5,
                           'Pleural Effusion': 6,
                           'Pneumothorax': 7,
                           'Endotracheal Tube': 8,
                           'Central Venous Line': 9,
                           'Monitoring Probes': 10,
                           'Nosogastric Tube': 11,
                           'Chest tube': 12,
                           'Tubings': 13}

        self.labels = pd.read_csv(os.path.join(path_to_data, "structure.csv"))
        self.fileNames = self.labels.Filename.unique()

    def __len__(self):
        """
        Returns:
            (int): length of image files in covid segmentation dataset
        """
        return len(self.fileNames)

    def __getitem__(self, idx):
        """
        Arguments:
        - idx (int) : Index of the image to return
        Returns:
        - image (PIL.Image): PIL format image
        - label: np.array with segmentation for each pixel
        """

        image_path = os.path.join(self.path_to_data, 'images', self.fileNames[idx])
        rows = self.labels[self.labels.Filename.eq(self.fileNames[idx])]
        image = Image.open(image_path).convert('RGB')
        label = np.zeros((image.size[1], image.size[0]))
        for index, row in rows.iterrows():
            value = row['Mask']
            if not isinstance(value, str):
                continue
            x_offset = row['Left']
            y_offset = row['Top']
            mask_path = os.path.join(self.path_to_data, value)
            tmp_mask = cv2.imread(mask_path, 0)
            _, tmp_mask = cv2.threshold(tmp_mask, 0, 1, cv2.THRESH_BINARY)
            tmp_label = np.zeros((image.size[1], image.size[0]))
            h, w = tmp_mask.shape
            width_offset = min(x_offset + w, image.size[0]) - x_offset
            height_offset = min(y_offset + h, image.size[1]) - y_offset
            tmp_label[y_offset:min(y_offset + h, image.size[1]), x_offset:min(x_offset + w, image.size[0])] = tmp_mask[
                                                                                                              0:height_offset,
                                                                                                              0:width_offset]
            pixel_value = self.PRED_LABEL[row['Template Name']] + 1

            label = np.maximum(np.array(tmp_label) * pixel_value, label)
        if self.output_label_size:
            label = cv2.resize(label, (56,56),  interpolation = cv2.INTER_NEAREST)
        # Calls the transforms on image
        if self.transform:
            image = self.transform(image)

        # Transform labels
        label = torch_transforms.ToTensor()(label.astype(int))
        return image, label


if __name__ == '__main__':
    from src.__init__ import ROOT_DIR

    transforms = None
    coviddat = CovidSeg(os.path.join(ROOT_DIR, 'covid-19-chest-xray-segmentations-dataset/'), transform=transforms, output_label_size=(224,224))
    for i in range(100):
        img, label = coviddat[i]
    # plt.imshow(label)
    # plt.axis('off')
    # plt.show()
