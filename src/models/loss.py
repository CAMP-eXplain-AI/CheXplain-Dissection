import torch
import sys

def global_mae(y_pred, y_target):
    """ Calculates global Mean Absolute Error (MAE)

    Arguments:
        - y_pred (torch.tensor): Predictions shape (B x 6 x 4)
        - y_target (torch.tensor): Targets shape (B x 6 x 4)
    Returns:
        - mae_loss (float): Calculated loss
    """

    score = torch.argmax(y_target, dim=2)
    global_score = torch.sum(score, dim=1).type(torch.float32)
    global_score_pred = torch.sum(torch.argmax(y_pred, dim=2), dim=1).type(torch.float32)
    g_mae = mae_2(global_score_pred, global_score)

    return g_mae

def scce(y_pred, y_target):
    """ Calculates Sparse Categorical Cross Entropy loss

    Arguments:
        - y_pred (torch.tensor): Predictions shape (B x 6 x 4)
        - y_target (torch.tensor): Targets shape (B x 6 x 4)
    Returns:
        - scce_loss (float): Calculated loss
    """

    num_preds = y_pred.shape[1] * y_pred.shape[0]

    log = torch.log(y_pred + sys.float_info.epsilon)
    scce_loss = torch.sum(torch.mul(log, y_target)) / num_preds

    return - scce_loss


def mae(y_pred, y_target):
    """ Calculates per zone Mean Absolute Error (MAE)

    Arguments:
        - y_pred (torch.tensor): Predictions shape (B x 6 x 4)
        - y_target (torch.tensor): Targets shape (B x 6 x 4)
    Returns:
        - mae_loss (float): Calculated loss
    """

    num_preds = y_pred.shape[1] * y_pred.shape[0]

    score = torch.argmax(y_target, dim=2)
    score_pred = torch.argmax(y_pred, dim=2)

    mae_loss = torch.sum(torch.abs(score - score_pred)) / num_preds

    return mae_loss

def weigthed_mse(y_pred, y_target, weights):
    """ Calculates weighted MSE loss

    Arguments:
        - y_pred (torch.tensor): Predictions shape (B x 1)
        - y_target (torch.tensor): Targets shape (B x 1)
        - weights (torch.tensor): Batch weights (B x 1)
    Returns:
        - mse_loss (float): Calculated weighted MSE loss
    """

    mse_loss = torch.mean(weights * (y_pred - y_target) ** 2)

    return mse_loss

def mse(y_pred, y_target):
    """ Calculates MSE loss

    Arguments:
        - y_pred (torch.tensor): Predictions shape (B x 1)
        - y_target (torch.tensor): Targets shape (B x 1)
    Returns:
        - mse_loss (float): Calculated weighted MSE loss
    """

    mse_loss = torch.mean((y_pred - y_target) ** 2)

    return mse_loss

def mae_2(y_pred, y_target):
    """ Calculates MAE loss for single target regression

    Arguments:
        - y_pred (torch.tensor): Predictions shape (B x 1)
        - y_target (torch.tensor): Targets shape (B x 1)
    Returns:
        - mse_loss (float): Calculated weighted MSE loss
    """

    mae_loss = torch.mean(torch.abs(y_pred - y_target))

    return mae_loss

