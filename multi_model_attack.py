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

'''
 这个文件用来生成图片[预测前],[攻击干扰],[攻击干扰后,单个模型的预测结果]
'''


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


def run(criterion, net_name, use_dataset_index, use_mask, save_img_pred: bool):
    assert net_name == "UNet" or net_name == "CENet" or net_name == "FCN"
    assert use_dataset_index < len(config.path_data)
    net = get_net(net_name).to(config.device)

    path_data = config.path_data[use_dataset_index]

    # 文件路径
    path_data_root = path_data["dataset"]
    path_checkpoints = path_data["checkpoints"]
    path_result = os.path.join(path_data["result"], net_name)
    os.makedirs(path_result, exist_ok=True)

    # 准备数据集
    dataset = MyDataset(path_data_root=path_data_root, phase="train", transform_list=transform_compose)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 创建文件夹
    path_img_pred, path_pert, path_pert_pred = create_result_dir(path_result,
                                                                 "pred", "pert", "pert_pred",
                                                                 config.checkpoints_indexes)

    # 冻结模型参数
    for param in net.parameters():
        param.requires_grad = False

    # M0~M9
    for index in config.checkpoints_indexes:
        # 载入已训练好的模型参数
        net.load_state_dict(torch.load(
            os.path.join(path_checkpoints, "{}_{}.pth").format(net_name, index),
            map_location=config.device))

        path_pert_save = os.path.join(path_pert, str(index))

        for epsilon in config.epsilon_list:
            for i, (img, label, mask, name) in enumerate(data_loader):
                img, label = img.to(config.device), label.to(config.device)
                label.requires_grad = False

                # 原图 -> 预测
                if save_img_pred:
                    pred_img = net(img.clone())[0]
                    pred_img[pred_img > 0] = 255
                    pred_img[pred_img < 255] = 0
                    path_save = os.path.join(path_img_pred, str(index), "{}.png".format(name[0]))
                    save_image(pred_img, path_save, normalize=False)

                img_cp = img.clone()
                img_cp.requires_grad = True

                # 定义干扰为 [480,480,1] 的零矩阵
                perturbation = torch.zeros(config.img_size).to(config.device)
                perturbation.requires_grad = True
                img_pert = img_cp + perturbation

                pred = net(img_pert)

                # if use_mask:
                #     mask = get_ring_range(label[0][0].data.cpu().numpy(), name[0])
                #     mask = torch.tensor(mask).to(config.device)
                #     loss = criterion(label * mask, pred * mask)
                # else:
                #     loss = criterion(label, pred)

                # 对loss进行反向传播 那么原本perturbation就携带了反传之后的grad
                loss = criterion(label, pred)
                loss.backward()

                # FGSM_attack
                # perturbation = perturbation.grad.data.sign() * epsilon
                perturbation = perturbation.grad.data.sign() * epsilon

                path_save_attacked = os.path.join(path_pert_save, "{}.npy".format(name[0]))
                np.save(path_save_attacked, perturbation.data.cpu().numpy())

        print("[%2d] 攻击图像生成！" % index)

        # 载入攻击后的数据集
        attacked_dataset = MyAttackedDataset(path_data_root=path_data_root, phase="train",
                                             path_pert=path_pert_save, transform_list=transform_compose)
        attacked_data_loader = DataLoader(attacked_dataset, batch_size=1, shuffle=False)

        # 载入模型 M0~M9
        for index_2 in config.checkpoints_indexes:
            net.load_state_dict(torch.load(
                os.path.join(path_checkpoints, "{}_{}.pth").format(net_name, index_2),
                map_location=config.device))

            for i, (img, label, mask, pert, name) in enumerate(attacked_data_loader):
                img, pert = img.to(config.device), pert.to(config.device)

                img_pert = img + pert
                pred_pert = net(img_pert)[0]
                pred_pert[pred_pert > 0] = 255
                pred_pert[pred_pert < 255] = 0
                path_save = os.path.join(path_pert_pred, str(index), str(index_2), "{}.png".format(name[0]))
                save_image(pred_pert, path_save, normalize=False)

        print("[%2d] 预测攻击图片完成！" % index)


if __name__ == "__main__":
    setup_seed(1)
    # run(nn.BCEWithLogitsLoss(), "UNet", 2, False, True)
    run(nn.BCEWithLogitsLoss(), "CENet", 0, False, True)
    # run(nn.BCEWithLogitsLoss(), "FCN", 2, False, True)
