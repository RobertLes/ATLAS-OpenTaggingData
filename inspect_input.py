#!/usr/bin/env python
# coding: utf-8

from utils import *
import numpy as np

import matplotlib.pyplot as plt
import time, sys
import argparse

if __name__ == '__main__':

  #Get input agrs
  parser = argparse.ArgumentParser()
  parser.add_argument("-d","--directory", default="./", help="The directory with the input files")
  args = parser.parse_args()

  #Get dataloaders, function in utils.py
  train_dataloader,test_dataloader=get_ATLAS_inputs(args.directory,"HL",batch_size=1)

  for x, y, z in train_dataloader:
    print(f"Shape of x: {x.shape}, {x.dtype}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    print(f"Shape of z: {z.shape} {z.dtype}")
    break

  #sanity plots
  Nplot=min(10000,len(train_dataloader.dataset))
  for var in range(len(train_dataloader.dataset[0][0])):
    sigvals=[]
    bkgvals=[]

    it = iter(train_dataloader)
    for batch, data in enumerate(train_dataloader):
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
    fig.savefig(f"input_distribution_{var}.pdf")
    plt.close(fig)
