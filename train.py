import os
import torch.nn as nn
import config
import torch
from dataset.Dataset import MyDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from model.UNet import UNet
from model.CENet import CENet
from model.FCN import *
from torch.optim import Adam
from utils.utils import setup_seed

'''
    训练文件
'''

transform_compose = transforms.Compose([transforms.Resize((480, 480)),
                                        transforms.ToTensor()])


def get_net(net_name):
    if net_name == "UNet":
        return UNet(3, 1)
    if net_name == "CENet":
        return CENet()
    if net_name == "FCN":
        vgg_model = VGGNet(requires_grad=True, show_params=False)
        return FCNs(pretrained_net=vgg_model, n_class=1)
    return None


def train(criterion, net_name, use_dataset_index):
    assert net_name == "UNet" or net_name == "CENet" or net_name == "FCN"
    assert use_dataset_index < len(config.path_data)
    path_data = config.path_data[use_dataset_index]

    path_data_root = path_data["dataset"]
    path_checkpoints = path_data["checkpoints"]
    os.makedirs(path_checkpoints, exist_ok=True)

    for seed in config.random_seed:
        # 设置随机种子
        setup_seed(seed)
        net = get_net(net_name).to(config.device)

        dataset = MyDataset(path_data_root=path_data_root, phase="train", transform_list=transform_compose)
        data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        optimizer = Adam(params=net.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        for epoch in range(config.epochs):
            for i, (img, label, mask, name) in enumerate(data_loader):
                img = img.to(config.device)
                label = label.to(config.device)

                pred = net(img)
                loss = criterion(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 50 == 0:
                    print("loss = {} ".format(loss))
            scheduler.step()

        torch.save(net.state_dict(), os.path.join(path_checkpoints, "{}_{}.pth").format(net_name, seed))


if __name__ == "__main__":
    criterion = nn.BCEWithLogitsLoss()
    train(criterion, "UNet", 3)
    # train(criterion, "FCN", 3)
    # train(criterion, "CENet", 3)

