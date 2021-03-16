import numpy as np


def get_acc(label, pred):
    temp = label == pred
    element_num = len(label.reshape(-1))
    acc = temp.sum() / element_num
    return acc


def get_iou(label, pred):
    intersection = np.logical_and(pred, label).sum()
    union = np.logical_or(pred, label).sum()
    return intersection / union


def get_dice(label, pred):
    inter = np.logical_and(label, pred).sum()  # 交集
    denominator = label.sum() + pred.sum()  # |X| + |Y|
    denominator = denominator + 1 if denominator == 0 else denominator  # 限制分母不能为零
    dice = 2 * inter / denominator
    return dice
