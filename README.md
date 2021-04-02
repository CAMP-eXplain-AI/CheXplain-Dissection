# Towards Semantic Interpretation of Thoracic Disease and COVID-19 Diagnosis Models
This is the source code for our paper. The contributions of this code are two-fold:
* Code for training and evaluation of COVID-19 models, mentioned in the paper.
* Network dissection setup for analysing the trained models. 


## Requirements
- pytorch 1.7
- torchvision 0.8.1
- tensorboard 2.4.0  
- opencv 4.0.1
- scikit-learn 0.23.2


## Training and evaluation 
### Data
The datasets [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/), 
[BrixIA](https://brixia.github.io/), 
[NIH ChestXray](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community), 
have to be manually downloaded and put in the ***data*** folder. This is because for CheXpert and BrixIA it is required to register to gain access to the data.
Under ***/src/data/*** two preprocessing functions are provided one for BrixIA and one for CheXpert.

- ***preprocess_brixia.py*** : 
  - Converts .dcm images to .jpg
  - Transforms images with torchvision transforms 
    

- ***preprocess_chexpert.py*** : 
  - Transforms images with torchvision transforms 
    
### Model training & evaluation

Functionalities to train and evaluate a model (DenseNet121) are in the ***main.py***.
The model is trained under a given config and afterwards evaluated. 
The model weights and [TensorBoard](https://www.tensorflow.org/tensorboard) log files are stored in the ***runs*** folder.

#### CheXpert model
A function to train and evaluate a model on the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) 
dataset on 14 pathologies with either [weighted BCE or unweighted BCE](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) 
is provided.

```python
chexpert_model(pretrained_model_path=None, **train_config)
```
Training parameters and experiment name can be defined in the ***train_config***, and a pretrained model can also be defined.
If no pretrained model is given, the pretrained weights on ImageNet are taken.
```python
pretrained_model_path = "./runs/chexpert_pretrained/model.pth"                  # Path to to a pretrained model.pth (DenseNet121) otherwise ImageNet weights are used

train_config = {
        'batch_size': 60,                                                       # Number of images per batch 
        'input_size': (224, 224),                                               # Image size of the model input
        'n_epochs': 20,                                                         # Number of max epochs
        'orientation': 'frontal',                                               # Only use the frontal images of CheXpert ('lateral', 'frontal', 'all)
        'optim': torch.optim.Adam,                                              # PyTorch optimizer
        'optim_kwargs': {'lr': 0.001, 'weight_decay': 0.0},                     # Optimizer parameters
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,                # PyTorch scheduler
        'scheduler_kwargs': {'factor': 0.1, 'patience': 3, 'mode': 'max'},      # Scheduler parameters
        'weighted_bce': False,                                                  # Use weighted or unweighted BCE
        'early_stopping': 10,                                                   # Stop training after n epochs if AUC does not improve
        'experiment_name': 'chexpert_model',                                    # Experiment name for runs folder
    }
```
#### BrixIA model
A function to train a model on the [BrixIA](https://brixia.github.io/) dataset on single target regression and six zone classification/regression.

```python
brixia_model(pretrained_model_path=None, **train_config)
```
Similar to before the training parameters can be specified in the ***train_config***. A previously trained CheXpert model can be used as pretrained model.
```python
pretrained_model_path = "./runs/chexpert_pretrained/model.pth"                  # Path to to a pretrained model.pth (DenseNet121) otherwise ImageNet weights are used

train_config = {
        'batch_size': 60,                                                       # Number of images per batch 
        'input_size': (224, 224),                                               # Image size of the model input
        'n_epochs': 20,                                                         # Number of max epochs
        'optim': torch.optim.Adam,                                              # PyTorch optimizer
        'optim_kwargs': {'lr': 0.001, 'weight_decay': 0.0},                     # Optimizer parameters
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,                # PyTorch scheduler
        'scheduler_kwargs': {'factor': 0.1, 'patience': 3, 'mode': 'max'},      # Scheduler parameters
        'early_stopping': 6,                                                    # Stop training after n epochs if AUC does not improve
        'experiment_name': 'brixia_model',                                      # Experiment name for runs folder
        'alpha': 0.7,                                                           # If mode=paper then alpha gives a balancing between the two loss functions for the six zone training
        'mode': 'paper',                                                        # 'paper' for six zone model, 'regression' for single target model
        'criterion': torch.nn.MSELoss(),                                        # If mode=regression the criterion specified here is used
    }
```
#### Combined model
A function to train a combined model on binary classification (Finding/NoFinding) on three datasets CheXpert, 
[NIH ChestXray](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community) 
and BrixIA.
The ***train_config*** is similar to the CheXpert config.
```python
combined_model(pretrained_model_path=None, **train_config)
```

## Network Dissection
You can analyze any of the trained models with network dissection to identify concepts that the network is looking for.

## Data
We used two datast to interpret the network results. You would need to download these datasets first, before running any network dissection analysis. 
* [COVID-19 segmentation dataset](https://github.com/GeneralBlockchain/covid-19-chest-xray-segmentations-dataset.git)
* [NIH ChestXray](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)

## Interpreting a model
You can see a sample analysis with network dissection of the BrixIA model in [Covid_seg_network_dissection_results notebook](https://github.com/CAMP-eXplain-AI/CheXplain-Dissection/blob/main/Covid_seg_network_dissection_results.ipynb)
You can replace the model defined there with a model of your choosing and run the network dissection analysis. Please refer to the explanations in the notebook to understand the resulst.
