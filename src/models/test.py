import torch
import os
from tqdm import tqdm
from src.__init__ import ROOT_DIR
from src.visualization.confusion_matrix import tensor_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from src.models.loss import scce, mae, mae_2, mse, global_mae


def test_chexpert(model, test_loader, **train_config):
    """ Tests a given model and saves results to tensorbaord

    Arguments:
        - model (torch.nn.Module): Pytorch model
        - test_loader (torch.utils.data.DataLoader): Data loader with test set
        - train_config (dict): Dictionary of train parameters
    Returns:
    """

    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Process running on: {device}")

    experiment_name = train_config['experiment_name']
    log_path = f"{ROOT_DIR}/runs/{experiment_name}"
    logger = SummaryWriter(log_path)

    os.makedirs(log_path, exist_ok=True)
    num_correct = 0
    num_examples = 0
    y_targets = []
    y_raw_preds = []
    y_preds = []

    model.to(device)
    model = model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            x, y_target = data
            x, y_target = x.to(device), y_target.to(device)
            y_pred = model(x)
            # Apply sigmoid to pred for metrics
            y_pred = torch.sigmoid(y_pred)
            y_raw_preds.extend(y_pred.cpu().tolist())

            y_pred[torch.where(y_pred > 0.5)] = 1.0
            y_pred[torch.where(y_pred <= 0.5)] = 0

            y_preds.extend(y_pred.cpu().tolist())
            y_targets.extend(y_target.cpu().tolist())
            num_correct += torch.sum(y_pred == y_target).item()
            num_examples += y_target.shape[0] * y_target.shape[1]

    test_accuracy = num_correct / num_examples

    # Compute precision, recall F1-score and support for test set
    test_prec, test_recall, test_f1, test_support = precision_recall_fscore_support(y_targets, y_preds,
                                                                                    average="macro")

    # Calculate test auc for each label
    y_targets = torch.tensor(y_targets)
    y_raw_preds = torch.tensor(y_raw_preds)
    auc = 0
    num_labels = 0
    for i, label in enumerate(test_loader.dataset.PRED_LABEL):
        try:
            auc_i = roc_auc_score(y_targets[:, i], y_raw_preds[:, i])
            auc += auc_i
            num_labels += 1
        except ValueError:
            auc_i = -1
        logger.add_scalar(f'CheXpert Labels/AUC {label}', auc_i)

    logger.add_scalar('CheXpert Test/Averaged Precision', test_prec)
    logger.add_scalar('CheXpert Test/Averaged Recall', test_recall)
    logger.add_scalar('CheXpert Test/Averaged F1_Score', test_f1)
    logger.add_scalar('CheXpert Test/Average AUC', auc/num_labels)
    logger.add_scalar('CheXpert Test/Accuracy', test_accuracy)
    logger.add_image('Test/Confusion Matrices',
                     tensor_confusion_matrix(y_pred, y_target, test_loader.dataset.PRED_LABEL))
    logger.close()

def test_brixia(model, test_loader, **train_config):
    """ Tests a given model and saves results to tensorbaord

    Arguments:
        - model (torch.nn.Module): Pytorch model
        - test_loader (torch.utils.data.DataLoader): Data loader with test set
        - train_config (dict): Dictionary of train parameters
    Returns:
    """

    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Process running on: {device}")

    experiment_name = train_config['experiment_name']
    mode = train_config.get('mode', 'paper')
    log_path = f"{ROOT_DIR}/runs/{experiment_name}"
    logger = SummaryWriter(log_path)

    os.makedirs(log_path, exist_ok=True)

    loss_scce = 0
    loss_mae = 0
    loss_mse = 0
    loss_mae_global = 0

    num_test_batches = len(test_loader)

    model.to(device)
    model = model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            x, y_target, _ = data
            x, y_target = x.to(device), y_target.to(device)

            if mode == 'paper':
                y_pred = model(x).reshape(-1, 6, 4)
                y_pred = torch.nn.Softmax(dim=2)(y_pred)

                loss_scce += scce(y_pred, y_target)
                loss_mae += mae(y_pred, y_target)
                loss_mae_global += global_mae(y_pred, y_target)

            elif mode == 'regression':
                y_pred = model(x)

                y_pred = (y_pred * test_loader.dataset.std + test_loader.dataset.mean)
                y_target = (y_target * test_loader.dataset.std + test_loader.dataset.mean)

                loss_mae += mae_2(y_pred, y_target)
                loss_mse += mse(y_pred, y_target)

    if mode == 'paper':
        loss_scce /= num_test_batches
        loss_mae /= num_test_batches
        loss_mae_global /= num_test_batches
        logger.add_scalar('BrixIA Test/Global MAE', loss_mae_global)
        logger.add_scalar('BrixIA Test/MAE', loss_mae)
        logger.add_scalar('BrixIA Test/SCCE', loss_scce)

    elif mode == 'regression':
        loss_mse /= num_test_batches
        loss_mae /= num_test_batches
        logger.add_scalar('BrixIA Test/MAE', loss_mae)
        logger.add_scalar('BrixIA Test/MSE', loss_mse)

    logger.close()


def test_combined(model, test_loader, **train_config):
    """ Tests the combined binary model and saves results to tensorbaord

    Arguments:
        - model (torch.nn.Module): Pytorch model
        - test_loader (torch.utils.data.DataLoader): Data loader with test set
        - train_config (dict): Dictionary of train parameters
    Returns:
    """

    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Process running on: {device}")

    experiment_name = train_config['experiment_name']
    log_path = f"{ROOT_DIR}/runs/{experiment_name}"
    logger = SummaryWriter(log_path)

    os.makedirs(log_path, exist_ok=True)
    num_correct = 0
    num_examples = 0
    y_targets = []
    y_raw_preds = []
    y_preds = []

    model.to(device)
    model = model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            x, y_target = data

            x, y_target = x.to(device), y_target.to(device)
            y_pred = model(x).squeeze()

            # Apply sigmoid to pred for metrics
            y_pred = torch.sigmoid(y_pred)
            y_raw_preds.extend(y_pred.cpu().tolist())

            y_pred[torch.where(y_pred > 0.5)] = 1.0
            y_pred[torch.where(y_pred <= 0.5)] = 0

            y_preds.extend(y_pred.cpu().tolist())
            y_targets.extend(y_target.cpu().tolist())
            num_correct += torch.sum(y_pred == y_target).item()
            num_examples += len(y_target)

    test_accuracy = num_correct / num_examples

    # Compute precision, recall F1-score and support for test set
    test_prec, test_recall, test_f1, test_support = precision_recall_fscore_support(y_targets, y_preds,
                                                                                    average="binary")

    # Calculate average auc
    y_targets = torch.tensor(y_targets)
    y_raw_preds = torch.tensor(y_raw_preds)

    test_auc = roc_auc_score(y_targets, y_raw_preds)

    logger.add_scalar('Combined Test/Averaged Precision', test_prec)
    logger.add_scalar('Combined Test/Averaged Recall', test_recall)
    logger.add_scalar('Combined Test/Averaged F1_Score', test_f1)
    logger.add_scalar('Combined Test/Average AUC', test_auc)
    logger.add_scalar('Combined Test/Accuracy', test_accuracy)
    logger.close()
