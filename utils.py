# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 10:24  2022-05-07

import torch


def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector.
    """
    diff = angle_difference(torch.argmax(y_true), torch.argmax(y_pred))
    return torch.mean(torch.cast(torch.abs(diff), torch.floatx()))


def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - torch.abs(torch.abs(x - y) - 180)


def angle_regression_loss(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a float number between 0 and 1.
    """
    return torch.mean(angle_difference(y_true * 360, y_pred * 360))
