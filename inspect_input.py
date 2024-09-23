#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torchinfo import summary

from weaver.nn.model.ParticleTransformer import ParticleTransformer

import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import argparse

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-n","--network", help="The type of network to run",choices=['HL', 'Constituent', 'Transformer'],required=True)
  args = parser.parse_args()

  #Get list of input files
  starttime=time.time()
  if args.network!="HL":
    traininglist=[]
    for i in range(10): traininglist.append(ATLASH5LowLevelDataset("./train_nominal_00"+str(i)+".h5",useTransformer=(args.network=="Transformer")))
    training_data = ConcatDataset(traininglist)
    test_data=ATLASH5LowLevelDataset("./test_nominal_000.h5",useTransformer=(args.network=="Transformer"))
  else:
    traininglist=[]
    for i in range(10): traininglist.append(ATLASH5HighLevelDataset("./train_nominal_00"+str(i)+".h5",transform=True))
    training_data = ConcatDataset(traininglist)
    test_data=ATLASH5HighLevelDataset("./test_nominal_000.h5",transform=True)
  print("Took %.3e seconds to run dataset"%((time.time()-starttime)))

  # Create data loaders.
  starttime=time.time()
  batch_size = 2**8
  train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
  test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
  print("Took %.3e seconds to run dataloader"%((time.time()-starttime)))

  for x, y, z in train_dataloader:
    print(f"Shape of x : {x.shape}, {x.dtype}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    print(f"Shape of z: {z.shape} {z.dtype}")
    #print(x,y,z)
    break

  #sanity plots
  Nplot=min(10000,len(training_data))
  for var in range(training_data[0][0].shape[0]):
    sigvals=[]
    bkgvals=[]
    #for jet in range(Nplot):
    #  if training_data[jet][1]==1:
    #    sigvals.append(training_data[jet][0][var].item())
    #  else:
    #    bkgvals.append(training_data[jet][0][var].item())
    it = iter(train_dataloader)
    #for jet in range(Nplot):
    for batch,data in enumerate(train_dataloader):
      for jet in range(len(data[0])):
        if data[1][jet]==1:
          sigvals.append(data[0][jet][var].item())
        else:
          bkgvals.append(data[0][jet][var].item())
      if batch*len(data[0])>Nplot: break

    fig, ax = plt.subplots()
    minval=min(np.min(sigvals),np.min(bkgvals))
    maxval=max(np.max(sigvals),np.max(bkgvals))
    bins=np.linspace(minval,maxval,20)

    #print("var %i mean %f std %f"%(var,np.mean(sigvals),np.std(sigvals)))
    #print("min %f max %f"%(minval,maxval))

    sigprob,*_=ax.hist(sigvals,bins=bins,histtype="step",density=True)
    bkgprob,*_=ax.hist(bkgvals,bins=bins,histtype="step",density=True)

    #metrics
    sigprob=np.asarray(sigprob,dtype=np.float32)
    bkgprob=np.asarray(bkgprob,dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'): #ignore the error warning from nan values
      S2=0.5*np.nansum((sigprob-bkgprob)**2/(sigprob+bkgprob))/len(sigprob) #TMVA Seperation
      Mprob=0.5*(sigprob+bkgprob)
      JSD=0.5*np.nansum(sigprob*np.log(sigprob/Mprob))+0.5*np.nansum(bkgprob*np.log(bkgprob/Mprob)) #Jensen Shannon Divergence
      TVD=np.max(abs(sigprob-bkgprob)) #Total variation distance, same as half the integreated differemce
      HD=np.sqrt(0.5*np.nansum((np.sqrt(sigprob)-np.sqrt(bkgprob))**2)/len(sigprob)) #Hellinger distance
    print(f"variable={var}, S2={S2}, JSD={JSD}, TVD={TVD}, HD={HD}")

    #ax.set_yscale('log')
    fig.savefig(f"sanity_{var}.pdf")
    plt.close(fig)
