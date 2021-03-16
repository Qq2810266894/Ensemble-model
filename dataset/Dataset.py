import os
import PIL.Image as Image
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, path_data_root, phase, transform_list=None):
        super(MyDataset, self).__init__()
        self.path_data_root = path_data_root
        self.transform_list = transform_list

        self.dir_img = os.path.join(self.path_data_root, 'image_data')  # 训练图像文件
        self.dir_label = os.path.join(self.path_data_root, 'label_data')  # 图像的结果文件夹
        self.dir_mask = os.path.join(self.path_data_root, 'mask_data')  # 图像的结果文件夹
        if phase:
            self.dir_img = os.path.join(self.dir_img, phase)
            self.dir_label = os.path.join(self.dir_label, phase)
            self.dir_mask = os.path.join(self.dir_mask, phase)

        # list image file names
        self.images = os.listdir(self.dir_img)
        # 除了后缀，label名称一样要与image一样。 且label的后缀一定要统一。
        label_name = os.listdir(self.dir_label)[0]
        label_suffix = label_name[label_name.index(".") + 1:]

        self.image_files = []
        self.label_files = []
        self.mask_files = []
        self.name = []
        for name in self.images:
            self.image_files.append(os.path.join(self.dir_img, name))
            name = name[:name.index(".")]
            self.label_files.append(os.path.join(self.dir_label, "{}.{}".format(name, label_suffix)))
            # self.mask_files.append(os.path.join(self.dir_mask, "{}.png".format(name)))
            self.name.append(name)

        # if not os.path.exists(self.dir_mask):
        #     os.makedirs(self.dir_mask, exist_ok=True)
        #     for name in self.name:
        #         path_label = os.path.join(self.dir_label, "{}.{}".format(name, label_suffix))
        #         mask = self.create_label_mask(path_label)
        #         mask.save(os.path.join(self.dir_mask, "{}.png".format(name)))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load image file
        img = Image.open(self.image_files[index])
        label = Image.open(self.label_files[index])
        # mask = Image.open(self.mask_files[index])
        name = self.name[index]
        if self.transform_list is not None:
            img = self.transform_list(img)
            label = self.transform_list(label)
            # mask = self.transform_list(mask)
        # label[label >= 0.5] = 1.
        # label[label < 1] = 0.
        # mask[mask >= 0.5] = 1.
        # mask[mask < 1] = 0.
        mask = 1
        return img, label, mask, name

    @staticmethod
    def create_label_mask(path_label, kernel_size=(4, 4), iterations=5):
        img = np.array(Image.open(path_label), dtype=np.uint8)
        img = cv2.Canny(img, 100, 200)
        kernel = np.ones(kernel_size, np.uint8)
        dilation = cv2.dilate(img, kernel, iterations=iterations)
        ret = Image.fromarray(dilation).convert("L")
        return ret


class MyAttackedDataset(Dataset):
    def __init__(self, path_data_root, phase, path_pert, transform_list=None):
        super(MyAttackedDataset, self).__init__()
        self.path_data_root = path_data_root
        self.path_pert = path_pert
        self.transform_list = transform_list

        self.dir_img = os.path.join(self.path_data_root, 'image_data')  # 训练图像文件
        self.dir_label = os.path.join(self.path_data_root, 'label_data')  # 图像的结果文件夹
        if phase:
            self.dir_img = os.path.join(self.dir_img, phase)
            self.dir_label = os.path.join(self.dir_label, phase)

        # list image file names
        self.images = os.listdir(self.dir_img)
        # 除了后缀，label名称一样要与image一样。 且label的后缀一定要统一。
        label_name = os.listdir(self.dir_label)[0]
        label_suffix = label_name[label_name.index(".") + 1:]

        self.image_files = []
        self.label_files = []
        self.label_mask_files = []
        self.pert_files = []
        self.name = []
        for name in self.images:
            self.image_files.append(os.path.join(self.dir_img, name))
            name = name[:name.index(".")]
            self.label_files.append(os.path.join(self.dir_label, "{}.{}".format(name, label_suffix)))
            # self.label_mask_files.append(os.path.join(self.dir_label_mask, "{}.bmp".format(name)))
            self.pert_files.append(os.path.join(self.path_pert, "{}.npy".format(name)))
            self.name.append(name)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load image file
        img = Image.open(self.image_files[index])
        label = Image.open(self.label_files[index])
        pert = torch.tensor(np.load(self.pert_files[index]))
        name = self.name[index]
        if self.transform_list is not None:
            img = self.transform_list(img)
            label = self.transform_list(label)
        label[label >= 0.5] = 1.
        label[label < 1] = 0.
        mask = 1
        return img, label, mask, pert, name
