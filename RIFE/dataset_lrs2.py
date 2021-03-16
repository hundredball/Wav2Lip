from torch.utils.data import DataLoader, Dataset
from intervaltree import IntervalTree
import glob
import os
import torch
from PIL import Image
import numpy as np


class LRS2Dataset(Dataset):
    def __init__(self, dataset_name, batch_size=32) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        path = '/teams/ECE285_WI21_A00/22/lrs2_preprocessed_full/main/'
        self.images_folders = sorted(glob.glob(os.path.join(path, '**/'), recursive=True))
        self.images_len = 0
        self.images = IntervalTree()
        self.flow = {}
        self.current_flow = None
        self.current_flow_name = ''
        for images_folder in self.images_folders:
            images = glob.glob(os.path.join(images_folder, '*.jpg'))
            images = [os.path.join(images_folder, f'{i}.jpg') for i in range(len(images))]
            images_len = len(images)
            if images_len > 2:
                images_len = images_len - 2
                self.images[self.images_len:self.images_len+images_len] = images
                self.images_len += images_len

        with open(os.path.join(path, 'path.txt'), 'r', encoding='utf-8') as pathfile:
            flow_filename = ''
            for index, line in enumerate(pathfile):
                line = line.strip()
                if index % 64 == 0:
                    flow_filename = line
                self.flow[line.replace('pt', 'jpg')] = (flow_filename, index % 64)

    def load_image(self, imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img

    def __len__(self):
        if self.dataset_name == 'validation':
            return 128
        return self.images_len

    def __getitem__(self, index):
        images_intervel = list(self.images[index])
        assert len(images_intervel) == 1
        images_intervel = images_intervel[0]
        image1_path = images_intervel.data[index - images_intervel.begin]
        image1 = self.load_image(image1_path)
        gt_path = images_intervel.data[index - images_intervel.begin + 1]
        gt = self.load_image(gt_path)
        image2_path = images_intervel.data[index - images_intervel.begin + 2]
        image2 = self.load_image(image2_path)

        current_flow_name, current_flow_index = self.flow[image1_path]
        if self.current_flow_name != current_flow_name:
            self.current_flow_name = current_flow_name
            self.current_flow = torch.load(self.current_flow_name)
        flow1 = self.current_flow[current_flow_index][2:]

        current_flow_name, current_flow_index = self.flow[gt_path]
        if self.current_flow_name != current_flow_name:
            self.current_flow_name = current_flow_name
            self.current_flow = torch.load(self.current_flow_name)
        flow2 = self.current_flow[current_flow_index][:2]
        flow_gt = torch.cat((flow1, flow2), 0)

        return torch.cat((image1, image2, gt), 0), flow_gt
