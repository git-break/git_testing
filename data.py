from torch.utils.data import Dataset
import os
import torch


class AOI_Data(Dataset):
    # img_type: PAN, MUL-PanSharpen, Both
    # dataset_type: train, val, test
    def __init__(self, dataset_root_path, img_type, dataset_type):
        super(AOI_Data, self).__init__()
        self.dataset_root_path = dataset_root_path
        self.img_type = img_type
        self.dataset_type = dataset_type
        if self.img_type != "Both":
            self.img_lis = [os.path.join(self.dataset_root_path, self.img_type, x) for x in
                            os.listdir(os.path.join(self.dataset_root_path, self.img_type))]
            # self.img_lis = os.listdir(os.path.join(self.dataset_root_path, self.img_type))
            self.label_lis = [os.path.join(self.dataset_root_path, 'mask_val1', x) for x in
                              os.listdir(os.path.join(self.dataset_root_path, 'mask_val1'))]
        else:
            self.img_lis = [os.path.join(self.dataset_root_path, 'PAN', x) for x in
                            os.listdir(os.path.join(self.dataset_root_path, 'PAN'))]
            self.img2_lis = [os.path.join(self.dataset_root_path, 'MUL', x) for x in
                             os.listdir(os.path.join(self.dataset_root_path, 'MUL'))]
            self.label_lis = [os.path.join(self.dataset_root_path, 'mask_val1', x) for x in
                              os.listdir(os.path.join(self.dataset_root_path, 'mask_val1'))]
        self.name_lis = []

    def __getitem__(self, idx):
        img_item_path = self.img_lis[idx]
        label_item_path = self.label_lis[idx]
        img = torch.load(img_item_path)
        if len(img.shape) == 2:
            img = img.view(1, img.shape[0], img.shape[1])
        label = torch.load(label_item_path)
        label = label.view(1, label.shape[0], label.shape[1])
        # print(img.shape)
        if self.img_type == 'Both':
            img2_item_path = self.img2_lis[idx]
            img2 = torch.load(img2_item_path)
            return img, img2, label, self.name_lis
        else:
            return img, label, self.name_lis

    def __len__(self):
        return len(self.img_lis)


class IEEE_Data(Dataset):
    # img_type: RGB, Hyper, Both
    # dataset_type: train, val, test
    def __init__(self, dataset_root_path, img_type, dataset_type):
        super(IEEE_Data, self).__init__()
        self.dataset_root_path = dataset_root_path
        self.img_type = img_type
        self.dataset_type = dataset_type
        if self.img_type != "Both":
            self.img_lis = [os.path.join(self.dataset_root_path, self.img_type, x) for x in
                            os.listdir(os.path.join(self.dataset_root_path, self.img_type))]
            # self.img_lis = os.listdir(os.path.join(self.dataset_root_path, self.img_type))
            self.label_lis = [os.path.join(self.dataset_root_path, 'GT', x) for x in
                              os.listdir(os.path.join(self.dataset_root_path, 'GT'))]
        else:
            self.img_lis = [os.path.join(self.dataset_root_path, 'RGB', x) for x in
                            os.listdir(os.path.join(self.dataset_root_path, 'RGB'))]
            self.img2_lis = [os.path.join(self.dataset_root_path, 'Hyper', x) for x in
                             os.listdir(os.path.join(self.dataset_root_path, 'Hyper'))]
            self.label_lis = [os.path.join(self.dataset_root_path, 'GT', x) for x in
                              os.listdir(os.path.join(self.dataset_root_path, 'GT'))]
        self.name_lis = []

    def __getitem__(self, idx):
        img_item_path = self.img_lis[idx]
        label_item_path = self.label_lis[idx]
        img = torch.load(img_item_path)
        label = torch.load(label_item_path)
        label = label.view(1, label.shape[0], label.shape[1])
        if self.img_type == 'Both':
            img2_item_path = self.img2_lis[idx]
            img2 = torch.load(img2_item_path)
            return img, img2, label, self.name_lis
        else:
            return img, label, self.name_lis

    def __len__(self):
        return len(self.img_lis)
