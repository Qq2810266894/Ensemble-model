import os
import torch

checkpoints_indexes = [i for i in range(50, 60)]
epsilon_list = [0.03]
img_size = (480, 480)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ######## train ########
random_seed = [i for i in range(50, 60)]
batch_size = 16
# batch_size = 28
lr = 0.001
epochs = 100

# ######## path ########
path = os.getcwd()
# path = "/home/pengzx/deepLearning/code/U-Net"
path_home = "/home/pengzx"

path_data_root = os.path.join(path_home, "dataset")

path_data_glaucoma = os.path.join(path_data_root, "Glaucoma")
path_data_glaucoma_mask = os.path.join(path_data_glaucoma, "label_mask")
path_data_isic = os.path.join(path_data_root, "ISIC")
path_data_isic_mask = os.path.join(path_data_isic, "label_mask")
path_data_cornea = os.path.join(path_data_root, "Cornea")
path_data_cornea_mask = os.path.join(path_data_cornea, "label_mask")
path_data_blood_vessel = os.path.join(path_data_root, "blood_vessel")
path_data_blood_vessel_mask = os.path.join(path_data_cornea, "label_mask")

path_checkpoints = os.path.join(path, "checkpoints")
path_checkpoints_glaucoma = os.path.join(path_checkpoints, "Glaucoma")
path_checkpoints_isic = os.path.join(path_checkpoints, "ISIC")
path_checkpoints_cornea = os.path.join(path_checkpoints, "Cornea")
path_checkpoints_blood_vessel = os.path.join(path_checkpoints, "blood_vessel")

path_result = os.path.join("/home/pengzx/deepLearning/", "result")
path_result_glaucoma = os.path.join(path_result, "Glaucoma")
path_result_isic = os.path.join(path_result, "ISIC")
path_result_cornea = os.path.join(path_result, "Cornea")
path_result_blood_vessel = os.path.join(path_result, "blood_vessel")

use_dataset = 2  # Glaucoma:0  ISIC:1  Cornea:2
path_data = []
path_data.append({
    "dataset": path_data_glaucoma, "checkpoints": path_checkpoints_glaucoma, "result": path_result_glaucoma
})
path_data.append({
    "dataset": path_data_isic, "checkpoints": path_checkpoints_isic, "result": path_result_isic
})
path_data.append({
    "dataset": path_data_cornea, "checkpoints": path_checkpoints_cornea, "result": path_result_cornea
})
path_data.append({
    "dataset": path_data_blood_vessel, "checkpoints": path_checkpoints_blood_vessel, "result": path_result_blood_vessel
})
