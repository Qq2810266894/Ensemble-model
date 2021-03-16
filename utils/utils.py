import os
import shutil
import numpy as np
import torch
import random


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_result_dir(path_result, name_pred, name_pert, name_pert_pred, indexes: list, clear_exist=False):

    path_pred = os.path.join(path_result, name_pred)
    path_pert = os.path.join(path_result, name_pert)
    path_pert_pred = os.path.join(path_result, name_pert_pred)

    if clear_exist:
        shutil.rmtree(path_pred, ignore_errors=True)
        shutil.rmtree(path_pert, ignore_errors=True)
        shutil.rmtree(path_pert_pred, ignore_errors=True)

    for i in indexes:
        os.makedirs(os.path.join(path_pred, str(i)), exist_ok=True)
        os.makedirs(os.path.join(path_pert, str(i)), exist_ok=True)
        for j in indexes:
            os.makedirs(os.path.join(path_pert_pred, str(i), str(j)), exist_ok=True)

    return path_pred, path_pert, path_pert_pred

