#!/usr/bin/env python

import torch
from torch import nn
from torch.utils.data import DataLoader,ConcatDataset
from torchinfo import summary

import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

from train import ATLASH5Dataset,NeuralNetwork

# Create data loaders.
testinglist=[]
for i in range(1): testinglist.append(ATLASH5Dataset("test_nominal_00"+str(i)+".h5",transform=True,return_pt=True))
test_data = ConcatDataset(testinglist)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

#Load model
device="cpu"
model = NeuralNetwork(test_data[0][0].shape[0]-1).to(device)
model.load_state_dict(torch.load("model.pth"))

Nbins=10
bins=np.linspace(0,3e6,Nbins+1,endpoint=True)
N_accepted=np.zeros(Nbins)
N_all=np.zeros(Nbins)

# find working point
with torch.no_grad():
  for X, y, w in test_dataloader:
    y=y.item()
    if y==0: continue #signal only

    pt=X[:,-1].item()
    X=X[:,:-1]

    pred=model(X)
    result=torch.argmax(pred, dim=1).item()

# make rejection
with torch.no_grad():
  for X, y, w in test_dataloader:
    y=y.item()
    if y==1: continue #background only

    pt=X[:,-1].item()
    X=X[:,:-1]

    pred=model(X)
    result=torch.argmax(pred, dim=1).item()
    #print(pred,result,y)

    #Save to rejection plot
    binx=int(np.floor(pt/(bins[1])))
    if binx>len(N_all)-1: continue
    N_all[binx]+=1
    N_accepted[binx]+=1
    if result==1: N_all[binx]+=1

fig, ax = plt.subplots()
ax.errorbar((bins[1:]+bins[:-1])/2, np.nan_to_num(N_all/N_accepted), xerr=(bins[1:]-bins[:-1])/2,fmt='o')
fig.savefig("rejection_pt.pdf")
plt.close(fig)
