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
from utils.evaluation import *

transform_compose = transforms.Compose([transforms.Resize((480, 480)), transforms.ToTensor()])


def run(criterion, net_name, use_dataset_index, use_mask, save_img_pred: bool):
    path_data = config.path_data[use_dataset_index]

    # 文件路径
    path_data_root = path_data["dataset"]
    path_checkpoints = path_data["checkpoints"]
    path_result = os.path.join(path_data["result"], net_name)
    os.makedirs(path_result, exist_ok=True)

    net = UNet().to(config.device)
    for param in net.parameters():
        param.requires_grad = False

    path_pert = "/home/pengzx/deepLearning/result/Glaucoma/UNet/pert/"

    iou_total = []
    for index in config.checkpoints_indexes:
        path_pert_save = os.path.join(path_pert, str(index))

        attacked_dataset = MyAttackedDataset(path_data_root=path_data_root, phase="train",
                                             path_pert=path_pert_save, transform_list=transform_compose)
        attacked_data_loader = DataLoader(attacked_dataset, batch_size=1, shuffle=False)

        for index_2 in config.checkpoints_indexes:
            net.load_state_dict(torch.load(
                os.path.join(path_checkpoints, "{}_{}.pth").format(net_name, index_2),
                map_location=config.device))

            iou_list = []
            for i, (img, label, mask, pert, name) in enumerate(attacked_data_loader):
                img, pert = img.to(config.device), pert.to(config.device)

                img_pert = img.clone() + pert

                pred = net(img_pert)
                pred[pred > 0] = 1.
                pred[pred < 1.] = 0.

                label[label > 0.5] = 1.
                label[label < 1.] = 0.
                iou = get_iou(pred[0].data.cpu().numpy(), label.data.cpu().numpy())
                iou_list.append(iou)
            iou_list = np.array(iou_list)
            print("模型[{}]产生的扰动,模型[{}]的预测结果iou={}", index, index_2, iou_list.mean())
            iou_total.append(iou_list)
    iou_total = np.array(iou_total)
    np.save("./iou_total.npy", iou_total)


if __name__ == "__main__":
    setup_seed(1)
    run(nn.BCEWithLogitsLoss(), "UNet", 0, False, True)