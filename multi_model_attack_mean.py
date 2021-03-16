import torch
from dataset.Dataset import MyDataset, MyAttackedDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from model.UNet import UNet
from model.CENet import CENet
from model.FCN import *
from utils.utils import *
from torchvision.utils import save_image
import cv2
import PIL.Image as Image
import config

transform_compose = transforms.Compose([transforms.Resize((480, 480)), transforms.ToTensor()])


def get_net(net_name):
    if net_name == "UNet":
        return UNet(3, 1)
    if net_name == "CENet":
        return CENet()
    if net_name == "FCN":
        vgg_model = VGGNet(requires_grad=True, show_params=False)
        return FCNs(pretrained_net=vgg_model, n_class=1)
    return None


def create_pert_mean(name, path_pert, path_pert_store):
    path_pert_save = os.path.join(path_pert_store, name + ".npy")
    if os.path.exists(path_pert_save):
        # 如果进入了这个if 代码之前以及运行过了 有了pert_mean这个文件了
        # 那么就直接读取
        return np.load(path_pert_save)
    # 反之就读取干扰  然后进行求mean
    dir_pert = os.listdir(path_pert)
    ret = []
    for dir_name in dir_pert:
        ret.append(np.load(os.path.join(path_pert, dir_name, name + ".npy")))
    ret = np.array(ret).mean(axis=0)
    np.save(os.path.join(path_pert_save), ret)
    return ret


# 看 multi_model_attack.py 的备注
def run(criterion, net_name, use_dataset_index, use_mask, save_img_pred: bool):
    assert net_name == "UNet" or net_name == "CENet" or net_name == "FCN"
    assert use_dataset_index < len(config.path_data)
    net = get_net(net_name).to(config.device)

    path_data = config.path_data[use_dataset_index]

    path_data_root = path_data["dataset"]
    path_checkpoints = path_data["checkpoints"]
    path_result = os.path.join(path_data["result"], net_name)
    os.makedirs(path_result, exist_ok=True)

    dataset = MyDataset(path_data_root=path_data_root, phase="train", transform_list=transform_compose)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    path_pert = os.path.join(path_result, "pert")
    path_pert_mean = os.path.join(path_result, "pert_mean")
    path_pert_mean_pred = os.path.join(path_result, "pert_mean_pred")
    for i in config.checkpoints_indexes:
        os.makedirs(os.path.join(path_pert_mean_pred, str(i)), exist_ok=True)
    os.makedirs(path_pert_mean, exist_ok=True)

    for param in net.parameters():
        param.requires_grad = False

    for index in config.checkpoints_indexes:
        net.load_state_dict(
            torch.load(os.path.join(path_checkpoints, "{}_{}.pth".format(net_name, index)),
                       map_location=config.device))

        for i, (img, label, mask, name) in enumerate(data_loader):
            name = name[0]
            img = img.to(config.device)
            pert_mean = create_pert_mean(name, path_pert, path_pert_mean)
            pert_mean = torch.tensor(pert_mean).to(config.device)
            img.requires_grad = False
            pert_mean.requires_grad = False

            img_pert = img + pert_mean

            pred_img = net(img_pert)
            pred_img[pred_img > 0] = 255
            pred_img[pred_img < 255] = 0
            path_save = os.path.join(path_pert_mean_pred, str(index), "{}.png".format(name))
            save_image(pred_img, path_save, normalize=False)


if __name__ == "__main__":
    # 设置随机种子
    setup_seed(1)
    # run(nn.BCEWithLogitsLoss(), "UNet", 2, False, True)
    run(nn.BCEWithLogitsLoss(), "CENet", 0, False, True)
    # run(nn.BCEWithLogitsLoss(), "FCN", 0, False, True)
