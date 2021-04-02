import pandas as pd
import sys
import os
import torch
import torchvision.transforms as torch_transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torch.utils.data import Dataset
import numpy as np
#sys.path.append('.')

class Nih_cropped(Dataset):

    def __init__(self, path_to_data, fold, transform=None):
        """ Dataset class for NHI (cropped BBoxes)

        Args:
            path_to_data (string): Path to dataset
            fold (string): train, valid, test, all
            transform (torchvision.Transforms): transforms performed on the samples
        """
        self.transform = transform
        self.path_to_data = path_to_data

        self.PRED_LABEL = [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltrate",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax"]

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

        image_path = os.path.join(self.path_to_data, self.labels.iloc[idx]['image'])
        image = Image.open(image_path).convert('RGB')

        # Get labels from the dataframe for current image
        label = self.labels.iloc[idx, :].loc[self.PRED_LABEL]
        label = label.to_numpy()

        # Calls the transforms on image
        if self.transform:
            image = self.transform(image)

        return image, label

class NIH_full(Dataset):
    def __init__(self, path_to_data, fold, transform=None):
        """ Dataset class for NHI full dataset

        Args:
            path_to_data (string): Path to dataset
            fold (string): train, valid, test, all
            transform (torchvision.Transforms): transforms performed on the samples
        """
        self.transform = transform
        self.path_to_data = path_to_data

        self.PRED_LABEL = [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltrate',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']

        # Loading the correct CSV for train, valid, test or all
        if fold == "train":
            self.labels = pd.read_csv(os.path.join(path_to_data, "full_train_fold.csv"))
        elif fold == "valid":
            self.labels = pd.read_csv(os.path.join(path_to_data, "full_val_fold.csv"))
        elif fold == "test":
            self.labels = pd.read_csv(os.path.join(path_to_data, "full_test_fold.csv"))
        elif fold == "all":
            self.labels = pd.read_csv(os.path.join(path_to_data, "full_train_fold.csv"))
            tmp1 = pd.read_csv(os.path.join(path_to_data, "full_val_fold.csv"))
            tmp2 = pd.read_csv(os.path.join(path_to_data, "full_test_fold.csv"))
            self.labels = self.labels.append(tmp1, ignore_index=True)
            self.labels = self.labels.append(tmp2, ignore_index=True)
        else:
            raise Exception("Wrong fold input given!")
    
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

        image_path = os.path.join(self.path_to_data, self.labels.iloc[idx]['image_name'])
        image = Image.open(image_path).convert('RGB')

        # Get labels from the dataframe for current image
        label = self.labels.iloc[idx, :].loc[self.PRED_LABEL]
        label = label.to_numpy()

        image_bbox = torch.empty(np.array(image).size)
        image_cropped = torch.empty(np.array(image).size)
        try:
            img_bbox_path = os.path.join(self.path_to_data, self.labels.iloc[idx]['bbox_image'])
            image_bbox = Image.open(img_bbox_path).convert('RGB')
            img_cropped_path = os.path.join(self.path_to_data, self.labels.iloc[idx]['bbox_cropped'])
            image_cropped = Image.open(img_cropped_path).convert('RGB')
            if self.transform:
                image_bbox = self.transform(image_bbox)
                image_cropped = self.transform(image_cropped)
        except Exception:
            pass

        # Calls the transforms on image
        if self.transform:
            image = self.transform(image)

        label = torch.from_numpy(label.astype(int))
        return image, label, image_bbox, image_cropped

class NIH_segmented:
    def __init__(self, path_to_data, fold, transform=None, output_label_size=False):
        """ Dataset class for NHI full dataset segmented

        Args:
            path_to_data (string): Path to dataset
            fold (string): train, valid, test, all
            transform (torchvision.Transforms): transforms performed on the samples
        """
        self.transform = transform
        self.path_to_data = path_to_data
        self.output_label_size = output_label_size

        self.PRED_LABEL = {
            'Atelectasis' : 0,
            'Cardiomegaly': 1,
            'Effusion' : 2,
            'Infiltrate' : 3,
            'Mass' : 4,
            'Nodule' : 5,
            'Pneumonia' : 6,
            'Pneumothorax' : 7,
            'Consolidation' : 8,
            'Edema' : 9,
            'Emphysema' : 10,
            'Fibrosis' : 11,
            'Pleural_Thickening' : 12,
            'Hernia' : 13}

        # Loading the correct CSV for train, valid, test or all
        if fold == "train":
            self.labels = pd.read_csv(os.path.join(path_to_data, "full_train_fold.csv"))
        elif fold == "valid":
            self.labels = pd.read_csv(os.path.join(path_to_data, "full_val_fold.csv"))
        elif fold == "test":
            self.labels = pd.read_csv(os.path.join(path_to_data, "full_test_fold.csv"))
        elif fold == "all":
            self.labels = pd.read_csv(os.path.join(path_to_data, "full_train_fold.csv"))
            tmp1 = pd.read_csv(os.path.join(path_to_data, "full_val_fold.csv"))
            tmp2 = pd.read_csv(os.path.join(path_to_data, "full_test_fold.csv"))
            self.labels = self.labels.append(tmp1, ignore_index=True)
            self.labels = self.labels.append(tmp2, ignore_index=True)
        else:
            raise Exception("Wrong fold input given!")

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
        - label (torch tensor): mask for bounding box in image
        """

        image_path = os.path.join(self.path_to_data, self.labels.iloc[idx]['image_name'])
        image = Image.open(image_path).convert('RGB')

        tmp_mask = np.zeros((image.size[0], image.size[1]))
        bbox_coords = np.empty(4)
        try:
            bbox_coords[0] = self.labels.iloc[idx]['bbox_x']
            bbox_coords[1] = self.labels.iloc[idx]['bbox_y']
            bbox_coords[2] = self.labels.iloc[idx]['bbox_w']
            bbox_coords[3] = self.labels.iloc[idx]['bbox_h']
            transform_bb = RescaleBB(224,1024)
            bbox_coords = transform_bb(bbox_coords)
            bbox_x, bbox_y, bbox_w, bbox_h = bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]
            tmp_mask[int(bbox_y):int(bbox_y + bbox_h), int(bbox_x):int(bbox_x + bbox_w)] =  1
            pixel_value = self.PRED_LABEL[self.labels.iloc[idx]['label']] + 1
            label = tmp_mask * pixel_value

            if self.output_label_size:
                label = cv2.resize(label, (56, 56),  interpolation = cv2.INTER_NEAREST)

        except Exception as e:
            print(e)

        # Calls the transforms on image
        if self.transform:
            image = self.transform(image)

        # Transform labels
        #label = torch.from_numpy(label.astype(int))
        label = torch_transforms.ToTensor()(label.astype(int))
        return image, label

class RescaleBB(object):
    """Rescale the bounding box in a sample to a given size.

    Args:
        output_image_size (int): Desired output size.
    """

    def __init__(self, output_image_size, original_image_size):
        assert isinstance(output_image_size, int)
        self.output_image_size = output_image_size
        self.original_image_size = original_image_size

    def __call__(self, sample):
        assert sample.shape == (4,)
        x, y, w, h = sample[0], sample[1], sample[2], sample[3]

        scale_factor = self.output_image_size / self.original_image_size
        new_x, new_y, new_w, new_h = x * scale_factor, y * scale_factor, w * scale_factor, h * scale_factor
        transformed_sample = np.array([new_x, new_y, new_w, new_h])

        return transformed_sample

if __name__ == '__main__':
    from src.__init__ import ROOT_DIR
    import torchvision.transforms as torch_transforms

    path_to_data = os.path.join(ROOT_DIR, 'data/nih/dataset')
    transforms = None

    img, label = NIH_segmented(path_to_data, 'test', transform=transforms)[1]
    #img.show()
    plt.imshow(label)
    plt.axis('off')
    plt.show()
