import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import statistics
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
import time
import numpy as np
from torchvision import datasets
from model import TemporalRegularityDetector
from dataloader import Hevi

parser = argparse.ArgumentParser()
#parser.add_argument('--save_model', type=str)
parser.add_argument('--root', type=str)
parser.add_argument('--gpu', type=str)
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def weights_init(m):
    if isinstance(m,nn.Conv2d):
    	nn.init.xavier_normal(m.weight.data)
    	nn.init.constant(m.bias.data,0.1)
    if isinstance(m,nn.ConvTranspose2d):
    	nn.init.xavier_normal(m.weight.data)
    	nn.init.constant(m.bias.data,0.1)
    if isinstance(m, nn.BatchNorm2d):
    	nn.init.normal(m.weight.data, mean=1.0, std=0.001)
    	nn.init.constant(m.bias.data, 0.001)

root = args.root
#save_model = args.save_model
dataset = Hevi(root)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True,num_workers=8)
model = TemporalRegularityDetector().apply(weights_init)
loss = nn.MSELoss()
if torch.cuda.is_available():
    model.cuda()
    loss.cuda()
optimizer = optim.Adagrad(model.parameters(),lr=0.01,weight_decay=0.0005)
model.train(True)
param = model
perror = 0
#list = ['model.conv_c1','model.conv_c2','model.conv_c3','model.deconv_d3','model.deconv_d2','model.deconv_d1','model.uppool_d3','model.uppool_d2','model.fin']
for epoch in range(1,101):
	totalLoss = 0.0
	start = time.time()
	#with torch.set_grad_enabled(training):
	for img in dataloader:
		bSize = img.shape[0]
		img = to_variable(img)
		output = model(img)
		Loss = loss(output,img)
		#regLoss = []
		#for layer in list:
		#	tmp = eval(layer)
		#	wNorm = np.linalg.norm(tmp.weight.data.cpu().numpy().flatten(),ord=2)
		#	regLoss.append(wNorm)
		#print(reconstLoss)
		#Norm2 = np.asarray(regLoss,dtype=np.float32)
		#Norm2 = 0.01*np.mean(Norm2)
		#Norm2 = np.asarray(Norm2,dtype=np.float32)
		#Norm2 = torch.from_numpy(Norm2)

		#cbLoss = 0.5*reconstLoss+to_variable(Norm2)
		#cbLoss = 0.5*reconstLoss.cpu().data[0]+Norm2
		#cbLoss = to_variable(cbLoss)
		optimizer.zero_grad()
		Loss.backward()
		optimizer.step()
		#print(Loss.cpu().data[0])
		totalLoss += Loss.cpu().data[0]*bSize
	end = time.time()
	if epoch%10 == 0:
		snapshot_path = './snapshots'
		if not os.path.isdir(snapshot_path):
			os.makedirs(snapshot_path)
		snapshot_name = 'epoch-{}-loss-{}.pth'.format(
                epoch,
                float("{:.2f}".format(totalLoss/len(dataloader.dataset)),
            ))
		torch.save(model.state_dict(), os.path.join(snapshot_path, snapshot_name))
	print('Epoch {:2}, | '
              'train loss : {:4.2f}, | '
              'running time: {:.2f} sec'.format(
                  epoch,
                  totalLoss/len(dataloader.dataset),
                  end-start,
              ))
