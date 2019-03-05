import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
import numpy as np
from torchvision.transforms import ToTensor
import os
import os.path
import cv2

def load_rgb_frames(image_dir):
    img = cv2.imread(image_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227,227), interpolation=cv2.INTER_AREA)
    return np.asarray(img,dtype=np.float32)

def make_dataset(root):
    dataset = []
    for videoName in os.listdir(root):
        videoFramePath = os.path.join(root,videoName)
        for frame in os.listdir(videoFramePath):
            path = os.path.join(videoFramePath,'',frame)
            dataset.append(path)
    return dataset

class Hevi(data_utl.Dataset):

    def __init__(self,root):
        self.data = make_dataset(root)
        self.root = root

    def __getitem__(self, index):
        path = self.data[index]
        img = load_rgb_frames(path)
        return ToTensor()(img)

    def __len__(self):
        return len(self.data)
