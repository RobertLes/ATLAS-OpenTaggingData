#!/usr/bin/env python

from utils import *
import torch
from weaver.nn.model.ParticleTransformer import ParticleTransformer

from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt
import sys,os,time
import argparse

if __name__ == '__main__':
	#Get input agrs
  parser = argparse.ArgumentParser()
  parser.add_argument("-n","--network", help="The type of network to run",choices=['HL', 'Constituent', 'Transformer'],required=True)
  parser.add_argument("-d","--directory", default="./", help="The directory with the input files")
  args = parser.parse_args()

  #Get dataloaders, function in utils.py
  train_dataloader,test_dataloader=get_ATLAS_inputs(args.directory,args.network,batch_size=256,return_pt=True)

  # Load model
  device="cpu"
  if args.network=="Constituent": Ndim = len(torch.flatten(torch.tensor(train_dataloader.dataset[0][0][0]))) #note extra index here from return_pt
  elif args.network=="Transformer": Ndim=len(train_dataloader.dataset[0][0][0])
  else: Ndim = len(train_dataloader.dataset[0][0][0])
  if args.network=="Transformer": model = ParticleTransformer(input_dim=Ndim,num_classes=2).to(device)
  else: model = DNNetwork(Ndim,useConstituents=(args.network=="Constituent")).to(device)

  outname="model_HL"
  if args.network=="Constituent": outname="model_const"
  if args.network=="Transformer": outname="model_transformer"
  model.load_state_dict(torch.load(outname+".pth"))

  #Load information into numpy arrays
  ys=[]
  preds_raw=[]
  weights=[]
  pts=[]
  with torch.no_grad():
    for (X, pt), y, w in test_dataloader:
      #Get prediction
      pred=model(X)
      if useBCE:
        pred=model(X).flatten()
        pred=torch.sigmoid(pred)
      if useNLL:
        pred=torch.exp(pred)
      else:
        pred=torch.sigmoid(pred)

      #append info
      preds_raw.append(pred.numpy())
      ys.append(y.numpy())
      pts.append(pt.numpy())
      weights.append(w.numpy())

  #Store things in useful per-jet format
  preds_raw=np.array(preds_raw)
  preds_raw=preds_raw.reshape(-1,preds_raw.shape[-1])
  ys=np.array(ys).flatten()
  pts=np.array(pts).flatten()
  weights=np.array(weights).flatten()

  #make confusion matrix
  confusion=metrics.confusion_matrix(ys, np.argmax(preds_raw,axis=1))
  print(confusion)

  #Decision rule
  preds=np.log(preds_raw[:,0])/np.log(preds_raw[:,1])

  #make ROC curve
  fpr, tpr, thresholds = metrics.roc_curve(ys, preds, sample_weight=weights)
  fig, ax = plt.subplots()
  ax.plot(tpr,1/fpr)
  ax.set_yscale('log')
  ax.set_xlim([0.3, 1.0])
  fig.savefig(f"ROC_{outname}.pdf")
  plt.close(fig)

  #Find 50% working point
  wp50=thresholds[np.argmax(tpr>0.5)]

  # make rejection plot at working point
  Nbins=10
  bins=np.linspace(0,3e6,Nbins+1,endpoint=True)
  N_accepted=np.zeros(Nbins)
  N_all=np.zeros(Nbins)

  bkg_indices=np.argwhere(ys==0)
  bkg_pts=pts[bkg_indices]
  bkg_preds=preds[bkg_indices]
  for ii in range(len(bkg_indices)):
    binx=int(np.floor(bkg_pts[ii]/(bins[1])))
    if binx>len(N_all)-1: continue

    N_all[binx]+=1
    if bkg_preds[ii]>wp50: N_accepted[binx]+=1

  fig, ax = plt.subplots()
  ax.errorbar((bins[1:]+bins[:-1])/2, np.nan_to_num(N_all/N_accepted), xerr=(bins[1:]-bins[:-1])/2,fmt='o')
  fig.savefig(f"rejection_pt_{outname}.pdf")
  plt.close(fig)
