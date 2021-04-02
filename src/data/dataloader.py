from src.features.dataset_chexpert import CheXPert
from src.features.dataset_brixia import BrixIA
from src.features.dataset_nih import Nih_cropped
from src.features.dataset_nih import NIH_full
from src.features.dataset_nih import NIH_segmented
from src.features.dataset_seg import CovidSeg
from src.features.dataset_combined import CombinedData
import torchvision.transforms as torch_transforms
import torch


def get_datalaoders(dataset, path_to_data, **train_config):
    """ Returns data loaders of given dataset

    Arguments:
        - dataset (string): 'chexpert', 'brixia', 'combined'
        - path_to_data (string): path to the dataset
        - train_config (dict): dictionary containing parameters
    Returns:
        - train_loader (torch.utils.data.DataLoader)
        - val_loader (torch.utils.data.DataLoader)
        - test_loader (torch.utils.data.DataLoader)
    """

    input_size = train_config['input_size']
    orientation = train_config.get('orientation', None)
    batch_size = train_config['batch_size']
    mode = train_config.get('mode', 'paper')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_workers = 8

    data_transforms = {
        'train': torch_transforms.Compose([
            torch_transforms.Resize(input_size),
            torch_transforms.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
            torch_transforms.RandomAffine(degrees=(-25, 25), translate=(0.15, 0.15), scale=(0.9, 1.1)),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean, std)
        ]),
        'val': torch_transforms.Compose([
            torch_transforms.Resize(input_size),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean, std)
        ]),
    }

    if dataset == 'chexpert':
        image_data = {
            'train': CheXPert(path_to_data=path_to_data,
                              fold='train',
                              transform=data_transforms['train'],
                              orientation=orientation),
            'val': CheXPert(path_to_data=path_to_data,
                            fold='valid',
                            transform=data_transforms['val'],
                            orientation=orientation),
            'test': CheXPert(path_to_data=path_to_data,
                             fold='test',
                             transform=data_transforms['val'],
                             orientation=orientation)
        }

    if dataset == 'brixia':
        image_data = {
            'train': BrixIA(path_to_data=path_to_data,
                            fold='train',
                            mode=mode,
                            transform=data_transforms['train']),
            'val': BrixIA(path_to_data=path_to_data,
                          fold='valid',
                          mode=mode,
                          transform=data_transforms['val']),
            'test': BrixIA(path_to_data=path_to_data,
                           fold='test',
                           mode=mode,
                           transform=data_transforms['val'])
        }

    if dataset == 'combined':
        image_data = {
            'train': CombinedData(path_to_nih=path_to_data[0],
                                  path_to_chexpert=path_to_data[1],
                                  path_to_brixia=path_to_data[2],
                                  fold='train',
                                  transform=data_transforms['train']),
            'val': CombinedData(path_to_nih=path_to_data[0],
                                path_to_chexpert=path_to_data[1],
                                path_to_brixia=path_to_data[2],
                                fold='valid',
                                transform=data_transforms['val']),
            'test': CombinedData(path_to_nih=path_to_data[0],
                                 path_to_chexpert=path_to_data[1],
                                 path_to_brixia=path_to_data[2],
                                 fold='test',
                                 transform=data_transforms['val'])
        }

    train_loader = torch.utils.data.DataLoader(
        image_data['train'],
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(
        image_data['val'],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        image_data['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_nih_cropped_dataloaders(path_to_data, **train_config):
    input_size = train_config['input_size']
    batch_size = train_config['batch_size']

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_workers = 8

    data_transforms = {
        'train': torch_transforms.Compose([
            torch_transforms.Resize(input_size),
            torch_transforms.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
            torch_transforms.RandomAffine(degrees=(-25, 25), translate=(0.15, 0.15), scale=(0.9, 1.1)),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean, std)
        ]),
        'val': torch_transforms.Compose([
            torch_transforms.Resize(input_size),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean, std)
        ]),
    }

    image_data = {
        'train': Nih_cropped(path_to_data=path_to_data,
                             fold='train',
                             transform=data_transforms['train']),
        'val': Nih_cropped(path_to_data=path_to_data,
                           fold='valid',
                           transform=data_transforms['val']),
        'test': Nih_cropped(path_to_data=path_to_data,
                            fold='test',
                            transform=data_transforms['val'])
    }

    nih_train_loader = torch.utils.data.DataLoader(
        image_data['train'],
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers)

    nih_val_loader = torch.utils.data.DataLoader(
        image_data['val'],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers)

    nih_test_loader = torch.utils.data.DataLoader(
        image_data['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    return nih_train_loader, nih_val_loader, nih_test_loader


def get_nih_full_dataloaders(path_to_data, **train_config):
    input_size = train_config['input_size']
    batch_size = train_config['batch_size']

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_workers = 8

    data_transforms = {
        'train': torch_transforms.Compose([
            torch_transforms.Resize(input_size),
            torch_transforms.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
            torch_transforms.RandomAffine(degrees=(-25, 25), translate=(0.15, 0.15), scale=(0.9, 1.1)),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean, std)
        ]),
        'val': torch_transforms.Compose([
            torch_transforms.Resize(input_size),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean, std)
        ]),
    }

    image_data = {
        'train': NIH_full(path_to_data=path_to_data,
                          fold='train',
                          transform=data_transforms['train']),
        'val': NIH_full(path_to_data=path_to_data,
                        fold='valid',
                        transform=data_transforms['val']),
        'test': NIH_full(path_to_data=path_to_data,
                         fold='test',
                         transform=data_transforms['val'])
    }

    nih_train_loader = torch.utils.data.DataLoader(
        image_data['train'],
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers)

    nih_val_loader = torch.utils.data.DataLoader(
        image_data['val'],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers)

    nih_test_loader = torch.utils.data.DataLoader(
        image_data['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    return nih_train_loader, nih_val_loader, nih_test_loader

def get_nih_segmented_dataloaders(path_to_data, **train_config):
    input_size = train_config['input_size']
    batch_size = train_config['batch_size']

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_workers = 8

    data_transforms = {
        'train': torch_transforms.Compose([
            torch_transforms.Resize(input_size),
            torch_transforms.ColorJitter(contrast=(0.8, 1.4), brightness=(0.8, 1.1)),
            torch_transforms.RandomAffine(degrees=(-25, 25), translate=(0.15, 0.15), scale=(0.9, 1.1)),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean, std)
        ]),
        'val': torch_transforms.Compose([
            torch_transforms.Resize(input_size),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean, std)
        ]),
    }

    image_data = {
        'train': NIH_segmented(path_to_data=path_to_data,
                     fold='train',
                     transform=data_transforms['train'],
                     output_label_size = input_size),
        'val': NIH_segmented(path_to_data=path_to_data,
                    fold='valid',
                    transform=data_transforms['val'],
                    output_label_size = input_size),
        'test': NIH_segmented(path_to_data=path_to_data,
                    fold='test',
                    transform=data_transforms['val'],
                    output_label_size = input_size)
        }

    nih_train_loader = torch.utils.data.DataLoader(
        image_data['train'],
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers)

    nih_val_loader = torch.utils.data.DataLoader(
        image_data['val'],
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers)

    nih_test_loader = torch.utils.data.DataLoader(
        image_data['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    return nih_train_loader, nih_val_loader, nih_test_loader

def get_segmentation_dataloader(path_to_data, **config):
    input_size = config['input_size']
    batch_size = config['batch_size']

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_workers = 2

    data_transforms = torch_transforms.Compose([
        torch_transforms.Resize(input_size),
        torch_transforms.ToTensor(),
        torch_transforms.Normalize(mean, std)])

    image_data = CovidSeg(path_to_data, transform=data_transforms, output_label_size=input_size)
    seg_loader = torch.utils.data.DataLoader(
        image_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
    return seg_loader


