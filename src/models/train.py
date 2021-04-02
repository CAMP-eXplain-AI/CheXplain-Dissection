import torch
import os
from src.__init__ import ROOT_DIR
from src.models.loss import scce, mae, weigthed_mse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def train_chexpert(model, train_loader, val_loader, **train_config):
    """
    Arguments:
        - model (torch.nn.Module): Pytorch model
        - train_loader (torch.utils.data.DataLoader): Data loader with training set
        - val_loader (torch.utils.data.DataLoader): Data loader with validation set
        - train_config (dict): Dictionary of train parameters
    Returns:
        - (string): Path of the trained model
    """

    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Process running on: {device}")

    early_stopping = train_config.get('early_stopping', None)
    experiment_name = train_config['experiment_name']
    n_epochs = train_config['n_epochs']
    criterion = train_config['criterion']
    optim = train_config['optim'](model.parameters(), **train_config['optim_kwargs'])
    scheduler = train_config['scheduler'](optim, **train_config['scheduler_kwargs'])
    log_path = f"{ROOT_DIR}/runs/{experiment_name}"
    logger = SummaryWriter(log_path)

    os.makedirs(log_path, exist_ok=True)

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    best_auc = None
    early_stopping_val_loss = None

    model.to(device)
    criterion.to(device)
    for i_epoch in range(n_epochs):

        epoch_train_loss = 0
        epoch_val_loss = 0

        model = model.train()
        for data in tqdm(train_loader):
            x, y_target = data

            optim.zero_grad()

            x, y_target = x.to(device), y_target.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y_target)
            loss.backward()
            optim.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= num_train_batches

        model = model.eval()
        num_correct = 0
        num_examples = 0
        y_target_list = []
        y_raw_pred_list = []
        y_pred_list = []

        with torch.no_grad():
            for data in tqdm(val_loader):
                x, y_target = data

                x, y_target = x.to(device), y_target.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y_target)
                epoch_val_loss += loss.item()

                # Apply sigmoid to pred for metrics
                y_pred = torch.sigmoid(y_pred)
                y_raw_pred_list.extend(y_pred.cpu().tolist())

                y_pred[torch.where(y_pred > 0.5)] = 1.0
                y_pred[torch.where(y_pred <= 0.5)] = 0

                y_pred_list.extend(y_pred.cpu().tolist())
                y_target_list.extend(y_target.cpu().tolist())
                num_correct += torch.sum(y_pred == y_target).item()
                num_examples += y_target.shape[0] * y_target.shape[1]

        epoch_val_loss /= num_val_batches
        epoch_val_accuracy = num_correct / num_examples

        # Compute precision, recall F1-score and support for validations set
        epoch_prec, epoch_recall, epoch_f1, epoch_support = precision_recall_fscore_support(y_target_list, y_pred_list,
                                                                                            average="macro")

        # Calculate average auc
        y_target_list = torch.tensor(y_target_list)
        y_raw_pred_list = torch.tensor(y_raw_pred_list)
        epoch_auc = 0
        num_labels = 0
        for i, label in enumerate(val_loader.dataset.PRED_LABEL):
            try:
                epoch_auc += roc_auc_score(y_target_list[:, i], y_raw_pred_list[:, i])
                num_labels += 1
            except ValueError:
                pass
        epoch_auc /= num_labels

        logger.add_scalar('CheXpert Validation/Precision', epoch_prec, i_epoch)
        logger.add_scalar('CheXpert Validation/Recall', epoch_recall, i_epoch)
        logger.add_scalar('CheXpert Validation/F1_Score', epoch_f1, i_epoch)
        logger.add_scalar('CheXpert Validation/AUC', epoch_auc, i_epoch)
        logger.add_scalar('CheXpert Validation/Accuracy', epoch_val_accuracy, i_epoch)
        logger.add_scalar('CheXpert Loss/Train', epoch_train_loss, i_epoch)
        logger.add_scalar('CheXpert Loss/Validation', epoch_val_loss, i_epoch)

        if best_auc is None or best_auc < epoch_auc:
            torch.save(model.state_dict(), f"{log_path}/model.pth")
            best_auc = epoch_auc

        if early_stopping is not None and i_epoch % early_stopping == 0:
            if early_stopping_val_loss != None and early_stopping_val_loss < epoch_val_loss:
                break
            else:
                # torch.save(model.state_dict(), f"{log_path}/checkpoint_{i_epoch}.pth")
                early_stopping_val_loss = epoch_val_loss

        scheduler.step(epoch_auc)

    logger.close()

    return f"{log_path}/model.pth"


def train_brixia(model, train_loader, val_loader, **train_config):
    """
    Arguments:
        - model (torch.nn.Module): Pytorch model
        - train_loader (torch.utils.data.DataLoader): Data loader with training set
        - val_loader (torch.utils.data.DataLoader): Data loader with validation set
        - train_config (dict): Dictionary of train parameters
    Returns:
        - (string): Path of the trained model
    """

    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Process running on: {device}")

    early_stopping = train_config.get('early_stopping', None)
    experiment_name = train_config['experiment_name']
    n_epochs = train_config['n_epochs']
    alpha = train_config.get('aplha', 0.5)
    optim = train_config['optim'](model.parameters(), **train_config['optim_kwargs'])
    scheduler = train_config['scheduler'](optim, **train_config['scheduler_kwargs'])
    log_path = f"{ROOT_DIR}/runs/{experiment_name}"
    logger = SummaryWriter(log_path)
    criterion = train_config.get('criterion', None)
    mode = train_config.get('mode', 'paper')

    os.makedirs(log_path, exist_ok=True)

    early_stopping_val_loss = None
    best_loss = None

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    model.to(device)
    for i_epoch in range(n_epochs):

        epoch_train_loss = 0
        epoch_val_loss = 0

        model = model.train()
        for data in tqdm(train_loader):
            x, y_target, weights = data
            x, y_target, weights = x.to(device), y_target.to(device), weights.to(device)
            optim.zero_grad()

            if mode == 'paper':
                y_pred = model(x).reshape(-1, 6, 4)
                y_pred = torch.nn.Softmax(dim=2)(y_pred)

                loss_scce = scce(y_pred, y_target)
                loss_mae = mae(y_pred, y_target)
                loss = alpha * loss_scce + (1 - alpha) * loss_mae

            elif mode == 'regression':
                y_pred = model(x)
                loss = criterion(y_pred, y_target)
                # loss = weigthed_mse(y_pred, y_target, weights)

            loss.backward()
            optim.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= num_train_batches

        model = model.eval()
        with torch.no_grad():
            for data in tqdm(val_loader):
                x, y_target, weights = data
                x, y_target, weights = x.to(device), y_target.to(device), weights.to(device)

                if mode == 'paper':
                    y_pred = model(x).reshape(-1, 6, 4)
                    y_pred = torch.nn.Softmax(dim=2)(y_pred)

                    loss_scce = scce(y_pred, y_target)
                    loss_mae = mae(y_pred, y_target)
                    loss = alpha * loss_scce + (1 - alpha) * loss_mae

                elif mode == 'regression':
                    y_pred = model(x)
                    loss = criterion(y_pred, y_target)
                    # loss = weigthed_mse(y_pred, y_target, weights)

                epoch_val_loss += loss.item()

            epoch_val_loss /= num_val_batches

        logger.add_scalar('Brixia Loss/Train', epoch_train_loss, i_epoch)
        logger.add_scalar('Brixia Loss/Validation', epoch_val_loss, i_epoch)

        if best_loss is None or best_loss > epoch_val_loss:
            torch.save(model.state_dict(), f"{log_path}/model.pth")
            best_loss = epoch_val_loss

        if early_stopping is not None and i_epoch % early_stopping == 0:
            if early_stopping_val_loss is not None and early_stopping_val_loss < epoch_val_loss:
                break
            else:
                # torch.save(model.state_dict(), f"{log_path}/checkpoint_{i_epoch}.pth")
                early_stopping_val_loss = epoch_val_loss

        scheduler.step(epoch_val_loss)

    logger.close()

    return f"{log_path}/model.pth"


def train_combined(model, train_loader, val_loader, **train_config):
    """
    Arguments:
        - model (torch.nn.Module): Pytorch model
        - train_loader (torch.utils.data.DataLoader): Data loader with training set
        - val_loader (torch.utils.data.DataLoader): Data loader with validation set
        - train_config (dict): Dictionary of train parameters
    Returns:
        - (string): Path of the trained model
    """

    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Process running on: {device}")

    early_stopping = train_config.get('early_stopping', None)
    experiment_name = train_config['experiment_name']
    n_epochs = train_config['n_epochs']
    criterion = train_config['criterion']
    optim = train_config['optim'](model.parameters(), **train_config['optim_kwargs'])
    scheduler = train_config['scheduler'](optim, **train_config['scheduler_kwargs'])
    log_path = f"{ROOT_DIR}/runs/{experiment_name}"
    logger = SummaryWriter(log_path)

    os.makedirs(log_path, exist_ok=True)

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    best_auc = None
    early_stopping_val_loss = None

    model.to(device)
    criterion.to(device)
    for i_epoch in range(n_epochs):

        epoch_train_loss = 0
        epoch_val_loss = 0

        model = model.train()
        for data in tqdm(train_loader):
            x, y_target = data

            optim.zero_grad()

            x, y_target = x.to(device), y_target.to(device)

            weights = torch.ones_like(y_target)
            weights.to(device)
            weights[y_target == 0] = 3.17

            y_pred = model(x).squeeze()
            loss = criterion(y_pred, y_target)
            loss = weights * loss
            loss = torch.mean(loss)
            loss.backward()
            optim.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= num_train_batches

        model = model.eval()
        num_correct = 0
        num_examples = 0
        y_target_list = []
        y_raw_pred_list = []
        y_pred_list = []

        with torch.no_grad():
            for data in tqdm(val_loader):
                x, y_target = data

                x, y_target = x.to(device), y_target.to(device)

                weights = torch.ones_like(y_target)
                weights.to(device)
                weights[y_target == 0] = 3.17

                y_pred = model(x).squeeze()
                loss = criterion(y_pred, y_target)
                loss = weights * loss
                loss = torch.mean(loss)
                epoch_val_loss += loss.item()

                # Apply sigmoid to pred for metrics
                y_pred = torch.sigmoid(y_pred)
                y_raw_pred_list.extend(y_pred.cpu().tolist())

                y_pred[torch.where(y_pred > 0.5)] = 1.0
                y_pred[torch.where(y_pred <= 0.5)] = 0

                y_pred_list.extend(y_pred.cpu().tolist())
                y_target_list.extend(y_target.cpu().tolist())
                num_correct += torch.sum(y_pred == y_target).item()
                num_examples += len(y_target)

        epoch_val_loss /= num_val_batches
        epoch_val_accuracy = num_correct / num_examples

        # Compute precision, recall F1-score and support for validations set
        epoch_prec, epoch_recall, epoch_f1, epoch_support = precision_recall_fscore_support(y_target_list, y_pred_list,
                                                                                            average="binary")
        # Calculate average auc
        y_target_list = torch.tensor(y_target_list)
        y_raw_pred_list = torch.tensor(y_raw_pred_list)

        epoch_auc = roc_auc_score(y_target_list, y_raw_pred_list)

        logger.add_scalar('Combined Validation/Precision', epoch_prec, i_epoch)
        logger.add_scalar('Combined Validation/Recall', epoch_recall, i_epoch)
        logger.add_scalar('Combined Validation/F1_Score', epoch_f1, i_epoch)
        logger.add_scalar('Combined Validation/AUC', epoch_auc, i_epoch)
        logger.add_scalar('Combined Validation/Accuracy', epoch_val_accuracy, i_epoch)
        logger.add_scalar('Combined Loss/Train', epoch_train_loss, i_epoch)
        logger.add_scalar('Combined Loss/Validation', epoch_val_loss, i_epoch)

        if best_auc is None or best_auc < epoch_auc:
            torch.save(model.state_dict(), f"{log_path}/model.pth")
            best_auc = epoch_auc

        if early_stopping is not None and i_epoch % early_stopping == 0:
            if early_stopping_val_loss != None and early_stopping_val_loss < epoch_val_loss:
                break
            else:
                # torch.save(model.state_dict(), f"{log_path}/checkpoint_{i_epoch}.pth")
                early_stopping_val_loss = epoch_val_loss

        scheduler.step(epoch_auc)

    logger.close()

    return f"{log_path}/model.pth"
