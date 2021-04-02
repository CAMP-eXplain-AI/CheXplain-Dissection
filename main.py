import os
import torch

from torchvision import models

from src.__init__ import ROOT_DIR
from src.data.dataloader import get_datalaoders
from src.models.test import test_chexpert
from src.models.train import train_chexpert
from src.models.test import test_brixia
from src.models.train import train_brixia
from src.models.train import train_combined
from src.models.test import test_combined

def chexpert_model(pretrained_model_path=None, **train_config):
    """ Trains a model on the CheXpert dataset

    Arguments:
        - pretrained_model_path (string): Path of the pretrained model
        - train_config (dict): Dictionary of train parameters
    Returns:
    """

    path_to_data = os.path.join(ROOT_DIR, 'data/chexpert/dataset')

    # Create data loaders
    train_loader, val_loader, test_loader = get_datalaoders('chexpert', path_to_data, **train_config)

    # Use weighted BCE
    if train_config['weighted_bce']:
        weights = train_loader.dataset.get_weights()
        train_config['criterion'] = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    else:
        train_config['criterion'] = torch.nn.BCEWithLogitsLoss()

    # Load pretrained model on ImageNet
    model = models.densenet121(pretrained=True)
    # Change last layer to CheXpert targets
    model.classifier = torch.nn.Linear(1024, len(train_loader.dataset.PRED_LABEL))

    # Load pretrained model on CheXpert
    if pretrained_model_path is not None:
        model.load_state_dict(torch.load(pretrained_model_path))

    # Train model
    pretrained_model_path = train_chexpert(model, train_loader, val_loader, **train_config)
    # Load trained model
    model.load_state_dict(torch.load(pretrained_model_path))
    # Test model
    test_chexpert(model, test_loader, **train_config)


def brixia_model(pretrained_model_path=None, **train_config):
    """ Trains a model on the BrixIA dataset

    Arguments:
        - pretrained_model_path (string): Path of the pretrained model
        - train_config (dict): Dictionary of train parameters
    Returns:
    """

    path_to_data = os.path.join(ROOT_DIR, 'data/brixia/')

    mode = train_config.get('mode', 'paper')
    if mode == 'paper':
        num_targets = 6 * 4
    elif mode == 'regression':
        num_targets = 1

    # Create data loaders
    train_loader, val_loader, test_loader = get_datalaoders('brixia', path_to_data, **train_config)

    # Load pretrained model on ImageNet
    model = models.densenet121(pretrained=True)

    if pretrained_model_path is not None:
        model_dict = torch.load(pretrained_model_path)
        # Get output shape of loaded model
        num_out = model_dict['classifier.weight'].shape[0]
        # Change last layer
        model.classifier = torch.nn.Linear(1024, num_out, bias=False)
        # Load pretrained model
        model.load_state_dict(model_dict)

    num_out = model.state_dict()['classifier.weight'].shape[0]
    # Change last layer to BrixIA targets
    if num_out != num_targets:
        model.classifier = torch.nn.Linear(1024, num_targets)

    # Train model
    pretrained_model_path = train_brixia(model, train_loader, val_loader, **train_config)
    # Load model
    model.load_state_dict(torch.load(pretrained_model_path))
    # Test model
    test_brixia(model, test_loader, **train_config)


def combined_model(pretrained_model_path=None, **train_config):
    """ Trains a model on the combined datasets

    Arguments:
        - pretrained_model_path (string): Path of the pretrained model
        - train_config (dict): Dictionary of train parameters
    Returns:
    """

    path_to_data = [os.path.join(ROOT_DIR, 'data/nih/dataset'),
                    os.path.join(ROOT_DIR, 'data/chexpert/dataset'),
                    os.path.join(ROOT_DIR, 'data/brixia/')]

    # Create data loaders
    train_loader, val_loader, test_loader = get_datalaoders('combined', path_to_data, **train_config)

    # Use weighted BC
    train_config['criterion'] = torch.nn.BCEWithLogitsLoss(reduction='none')

    # Load pretrained model on ImageNet
    model = models.densenet121(pretrained=True)
    # Change last layer to binary classification
    model.classifier = torch.nn.Linear(1024, 1)

    # Load pretrained model
    if pretrained_model_path is not None:
        model.load_state_dict(torch.load(pretrained_model_path))

    # Train model
    pretrained_model_path = train_combined(model, train_loader, val_loader, **train_config)
    # Load trained model
    model.load_state_dict(torch.load(pretrained_model_path))
    # Test model
    test_combined(model, test_loader, **train_config)


if __name__ == '__main__':

    pre_path = "./runs/chexpert_pretrained/model.pth"

    train_config = {
        'batch_size': 60,
        'input_size': (224, 224),
        'n_epochs': 20,
        'orientation': 'frontal',
        'optim': torch.optim.Adam,
        'weighted_bce': False,
        'optim_kwargs': {'lr': 0.001, 'weight_decay': 0.0},
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_kwargs': {'factor': 0.1, 'patience': 3, 'mode': 'max'},
        'early_stopping': 10,
        'experiment_name': 'chexpert_4_noweights',
    }
    chexpert_model(**train_config)

    train_config = {
        'batch_size': 60,
        'input_size': (224, 224),
        'n_epochs': 30,
        'optim': torch.optim.Adam,
        'optim_kwargs': {'lr': 0.0001, 'weight_decay': 0.0},
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_kwargs': {'factor': 0.1, 'patience': 4, 'mode': 'max'},
        'alpha': 0.7,
        'early_stopping': 6,
        'mode': 'paper',
        'criterion': torch.nn.MSELoss(),
        'experiment_name': 'brixia_checkpoints',
    }

    #brixia_model("./runs/chexpert_3_continue_noweights", **train_config)

    train_config = {
        'batch_size': 60,
        'input_size': (224, 224),
        'n_epochs': 20,
        'orientation': 'frontal',
        'optim': torch.optim.Adam,
        'optim_kwargs': {'lr': 0.001, 'weight_decay': 0.0},
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_kwargs': {'factor': 0.1, 'patience': 3, 'mode': 'max'},
        'early_stopping': 10,
        'experiment_name': 'combined_1',
    }

    #combined_model(**train_config)