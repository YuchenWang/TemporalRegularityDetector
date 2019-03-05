import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
import numpy as np
import os
import os.path
import cv2
import argparse
from torch.autograd import Variable
import torchvision
from torchvision.transforms import ToTensor
import time
from torchvision import datasets
from model import TemporalRegularityDetector
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--load_model', type=str)
parser.add_argument('--root', type=str)
parser.add_argument('--gpu', type=str)
parser.add_argument('--saved_path', type=str)
parser.add_argument('--pkl_path', type=str)

args = parser.parse_args()
anan_label_pkl = args.pkl_path
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
root = args.root
saved_path = args.saved_path
load_model = args.load_model

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def load_rgb_frames(image_dir):
    img = cv2.imread(image_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227,227), interpolation=cv2.INTER_AREA)
    return np.asarray(img,dtype=np.float32)

def make_dataset(root,pkl):
    dataset = []
    with open(pkl, 'rb') as dic:
        datasetDic = pickle.load(dic)
    #print(len(datasetDic))
    #del datasetDic['_000000']
    for clip in datasetDic.keys():
        video_name = datasetDic[clip]['video_name']
        idx = datasetDic[clip]['idx']
        clip_start = int(datasetDic[clip]['clip_start'])
        clip_end = int(datasetDic[clip]['clip_end'])
        target = datasetDic[clip]['target']
        for index in range(clip_start,clip_end+1):
            label = target[index - clip_start]
            dataset.append((video_name,idx,index,label))
    return dataset

class Anan(data_utl.Dataset):

    def __init__(self,root,pkl):

        self.data = make_dataset(root,pkl)
        self.root = root

    def __getitem__(self, index):
        video_name,clip,frameId,label= self.data[index]
        img = load_rgb_frames(os.path.join(self.root,video_name,str(frameId).zfill(6)+'.jpg'))
        return ToTensor()(img),label,video_name+'_'+clip+'_'+str(frameId).zfill(6)

    def __len__(self):
        return len(self.data)

def calculate_score(predict,groudTruth):
    error = np.power((predict-groudTruth),2)
    loss = np.sum(error)
    score = 1-(loss-np.min(error))/(np.max(error))
    return score

dataset = Anan(root,anan_label_pkl)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=1)
model = TemporalRegularityDetector()
model.load_state_dict(torch.load(load_model))
if torch.cuda.is_available():
    model.cuda()
model.train(False)
anan_data_dic = {}
count = 0
start = time.time()
for img,label,key in dataloader:
        gt = img
        img = to_variable(img)
        output = model(img)
        gt = gt.numpy()
        predict = output.data.cpu().numpy()
        score = calculate_score(predict,gt)
        anan_data_dic[key] = {
        'label':label,
        'score':score,
        }
        count = count +1
        if count%1000 ==0:
            current = time.time()
            print('Count {:2},|' 'running time:{:.2f} sec'.format(count,current-start))
with open(saved_path,'wb') as pk:
    pickle.dump(anan_data_dic,pk)
    print('Write pickle file to {}'.format(saved_path))
pk.close()