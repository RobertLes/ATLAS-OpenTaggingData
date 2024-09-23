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

useNLL=False
useBCE=False

class ATLASH5LowLevelDataset(torch.utils.data.Dataset):
  #copied from Kevin Grief https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data/-/blob/master/utils.py?ref_type=heads#L21
  def transform(self,index,max_constits):
    # Pull data 
    pt = np.asarray([self.clus_pt[index][:max_constits]]) #NOTE this is changed w.r.t Kevin
    eta = np.asarray([self.clus_eta[index][:max_constits]])
    phi = np.asarray([self.clus_phi[index][:max_constits]])
    energy = np.asarray([self.clus_E[index][:max_constits]])

    # Find location of zero pt entries in each jet. This will be used as a
    # mask to re-zero out entries after all preprocessing steps
    mask = np.asarray(pt == 0).nonzero()

    ########################## Angular Coordinates #############################

    # 1. Center hardest constituent in eta/phi plane. First find eta and
    # phi shifts to be applied
    eta_shift = eta[:,0]
    phi_shift = phi[:,0]

    # Apply them using np.newaxis
    eta -= eta_shift[:,np.newaxis]
    phi -= phi_shift[:,np.newaxis]

    # Fix discontinuity in phi at +/- pi using np.where
    phi = np.where(phi > np.pi, phi - 2*np.pi, phi)
    phi = np.where(phi < -np.pi, phi + 2*np.pi, phi)

    # 2. Rotate such that 2nd hardest constituent sits on negative phi axis
    second_eta = eta[:,1]
    second_phi = phi[:,1]
    alpha = np.arctan2(second_phi, second_eta) + np.pi/2
    eta_rot = (eta * np.cos(alpha[:,np.newaxis]) +
               phi * np.sin(alpha[:,np.newaxis]))
    phi_rot = (-eta * np.sin(alpha[:,np.newaxis]) +
               phi * np.cos(alpha[:,np.newaxis]))
    eta = eta_rot
    phi = phi_rot

    # 3. If needed, reflect so 3rd hardest constituent is in positive eta
    third_eta = eta[:,2]
    parity = np.where(third_eta < 0, -1, 1)
    eta = (eta * parity[:,np.newaxis]).astype(np.float32)
    # Cast to float32 needed to keep numpy from turning eta to double precision

    # 4. Calculate R with pre-processed eta/phi
    radius = np.sqrt(eta ** 2 + phi ** 2)

    ############################# pT and Energy ################################

    # Take the logarithm, ignoring -infs which will be set to zero later
    log_pt = np.log(pt)
    log_energy = np.log(energy)

    # Sum pt and energy in each jet
    sum_pt = np.sum(pt, axis=1)
    sum_energy = np.sum(energy, axis=1)

    # Normalize pt and energy and again take logarithm
    lognorm_pt = np.log(pt / sum_pt[:,np.newaxis])
    lognorm_energy = np.log(energy / sum_energy[:,np.newaxis])

    ########################### Finalize and Return ############################

    # Reset all of the original zero entries to zero
    eta[mask] = 0
    phi[mask] = 0
    log_pt[mask] = 0
    log_energy[mask] = 0
    lognorm_pt[mask] = 0
    lognorm_energy[mask] = 0
    radius[mask] = 0

    # Stack along last axis
    features = [eta, phi, log_pt, log_energy,
                lognorm_pt, lognorm_energy, radius]
    stacked_data = np.stack(features, axis=-1)

    return stacked_data

  def __init__(self, file_path, NConstituents=50, useTransformer=False, return_pt=False):
    super(ATLASH5LowLevelDataset, self).__init__()
    h5_file = h5py.File(file_path , 'r')
    self.data=torch.tensor([])

    self.clus_E=h5_file['fjet_clus_E']
    self.clus_eta=h5_file['fjet_clus_eta']
    self.clus_phi=h5_file['fjet_clus_phi']
    self.clus_pt=h5_file['fjet_clus_pt']
    self.clus_tast=h5_file['fjet_clus_taste']

    self.pt = h5_file['fjet_pt']
    self.labels = h5_file['labels']

    if "training_weights" in h5_file:
      self.hasWeights=True
      self.weights = h5_file['training_weights']
    else:
      self.hasWeights=False

    self.return_pt = return_pt
    self.NConstituents = NConstituents
    self.useTransformer = useTransformer

  def __getitem__(self, index):
    self.data=torch.squeeze(torch.tensor(self.transform(index,self.NConstituents)),dim=0)
    if self.useTransformer: self.data=torch.swapaxes(self.data, 0, 1)

    if self.return_pt:
      self.data=torch.cat((self.data,torch.tensor([self.pt[index]])))

    if self.hasWeights:
      return self.data,torch.tensor(self.labels[index],dtype=torch.int64),torch.tensor(self.weights[index])
    else:
      return self.data,torch.tensor(self.labels[index],dtype=torch.int64),torch.tensor(1)

  def __len__(self):
    return len(self.labels)


class ATLASH5HighLevelDataset(torch.utils.data.Dataset):
  def transform(self,data,transform=True):
    if transform:
      #Calculate some metrics from subsample of total
      vals=[]
      Njets=np.max([1000,data.len()])
      for jet in range(Njets): vals.append(data[jet])
      maxval=np.max(vals)
      minval=np.min(vals)
      return (data-minval)/(maxval-minval)

      #means=np.mean(vals)
      #stds=np.std(vals)
      #return (data-means)/stds
    else:
      return data

  def __init__(self, file_path, transform=True, return_pt=False):
    super(ATLASH5HighLevelDataset, self).__init__()
    h5_file = h5py.File(file_path , 'r')
    self.data=torch.tensor([])

    self.C2=self.transform(h5_file['fjet_C2'],transform=transform)
    self.D2=self.transform(h5_file['fjet_D2'],transform=transform)
    self.ECF1=self.transform(h5_file['fjet_ECF1'],transform=transform)
    self.ECF2=self.transform(h5_file['fjet_ECF2'],transform=transform)
    self.ECF3=self.transform(h5_file['fjet_ECF3'],transform=transform)
    self.L2=self.transform(h5_file['fjet_L2'],transform=transform)
    self.L3=self.transform(h5_file['fjet_L3'],transform=transform)
    self.Qw=self.transform(h5_file['fjet_Qw'],transform=transform)
    self.Split12=self.transform(h5_file['fjet_Split12'],transform=transform)
    self.Split23=self.transform(h5_file['fjet_Split23'],transform=transform)
    self.Tau1_wta=self.transform(h5_file['fjet_Tau1_wta'],transform=transform)
    self.Tau2_wta=self.transform(h5_file['fjet_Tau2_wta'],transform=transform)
    self.Tau3_wta=self.transform(h5_file['fjet_Tau3_wta'],transform=transform)
    self.Tau4_wta=self.transform(h5_file['fjet_Tau4_wta'],transform=transform)
    self.ThrustMaj=self.transform(h5_file['fjet_ThrustMaj'],transform=transform)
    self.m=self.transform(h5_file['fjet_m'],transform=transform)

    self.pt = h5_file['fjet_pt']
    self.labels = h5_file['labels']

    if "training_weights" in h5_file:
      self.hasWeights=True
      self.weights = h5_file['training_weights']
    else: 
      self.hasWeights=False

    self.return_pt = return_pt

  def __getitem__(self, index): 
    self.data=torch.tensor([self.D2[index],self.C2[index],self.ECF1[index],self.ECF2[index],self.ECF3[index],self.L2[index],self.L3[index],self.Qw[index],self.Split12[index],self.Split23[index],self.Tau1_wta[index],self.Tau2_wta[index],self.Tau3_wta[index],self.Tau4_wta[index],self.ThrustMaj[index],self.m[index]])

    if self.return_pt:
      self.data=torch.cat((self.data,torch.tensor([self.pt[index]])))

    if self.hasWeights:
      return self.data,torch.tensor(self.labels[index],dtype=torch.int64),torch.tensor(self.weights[index])
    else:
      return self.data,torch.tensor(self.labels[index],dtype=torch.int64),torch.tensor(1)

  def __len__(self):
    return len(self.labels)

class DNNetwork(nn.Module):
  def __init__(self,Ninputs,useConstituents=False):
    super().__init__()
    self.useConstituents=useConstituents

    if self.useConstituents: self.flatten = nn.Flatten()
    #self.norm=nn.BatchNorm1d(Ninputs)
    self.fc1= nn.Linear(Ninputs, 512)
    self.act1=nn.ReLU()
    self.fc2= nn.Linear(512, 512)
    self.act2=nn.ReLU()
    self.fc3= nn.Linear(512, 512)
    self.act3=nn.ReLU()
    self.fc4= nn.Linear(512, 2)
    if useBCE: self.fc4= nn.Linear(512, 1)
    if useNLL: self.act4=nn.LogSoftmax(dim=1)

  def forward(self, x):
    if self.useConstituents: x = self.flatten(x)
    #x=self.norm(x)
    f1=self.act1(self.fc1(x))
    f2=self.act2(self.fc2(f1))
    f3=self.act3(self.fc3(f2))
    logits = self.fc4(f3)
    if useBCE: return logits
    if useNLL: return self.act3(logits)
    else: return logits

def train(dataloader, model, loss_fn, optimizer, loss_train):
  #set to train mode
  model.train()

  #Loop over batches
  size = len(dataloader.dataset)
  for batch, (X, y, w) in enumerate(dataloader):
    #if(next(model.parameters()).is_cuda): dev=next(model.parameters()).get_device() 
    dev=device
    X, y, w = X.to(dev), y.to(dev), w.to(dev)

    # Compute loss
    pred = model(X)
    if useBCE: 
      pred=pred.flatten()
      y=y.float()
    loss = loss_fn(pred, y)
    loss = (loss * w / w.sum()).sum()

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Print some status
    if batch % 100 == 0:
      lossval, current = loss.item(), (batch + 1) * len(X)
      print(f"Training loss: {lossval:>7f}  [{current:>5d}/{size:>5d}]")

  loss_train.append(loss.item())

def test(dataloader, model, loss_fn, loss_test, acc_test):
  #Set to eval mode
  model.eval()

  #Loop over batches
  num_samples = len(dataloader.dataset)
  num_batches = len(dataloader)
  avgloss, acc = 0, 0
  with torch.no_grad():
    for X, y, w in dataloader:
      X, y, w = X.to(device), y.to(device), w.to(device)

      #compute loss
      pred=model(X)
      if useBCE:
        pred=model(X).flatten()
        y=y.float()
      loss=loss_fn(pred, y)
      avgloss += loss.sum().item()

      #compute accuracy
      if useNLL: pred=torch.exp(pred)
      else: pred=torch.sigmoid(pred)
      if useBCE:
        pred=torch.sigmoid(pred)
        acc += (pred.round() == y.int()).type(torch.float).sum().item()
      else:
        acc += (torch.argmax(pred, dim=1) == y).type(torch.float).sum().item()

  #save and print some info
  avgloss /= num_samples
  acc /= num_samples
  loss_test.append(avgloss)
  acc_test.append(acc)
  print(f"Testing Error: Accuracy: {(100*acc):>0.1f}%, Avg loss: {avgloss:>8f}")

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
  '''
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
  sys.exit()
  '''

  # Get cpu, gpu or mps device for training.
  starttime=time.time()
  device = ( "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
  if device=="cuda":
    print(f"Using device {device,torch.cuda.get_device_name(0)}")
  else:
    print(f"Using device \"{device}\"")

  # Define model
  if args.network!="HL": Ndim = torch.flatten(x[0]).size(dim=0)
  else: Ndim = x.size(dim=1)
  if args.network=="Transformer": model = ParticleTransformer(input_dim=x.size(dim=1),num_classes=2).to(device)
  else: model = DNNetwork(Ndim,useConstituents=(args.network=="Constituent")).to(device)

  '''
  if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    #model = nn.parallel.DistributedDataParallel(model)
  '''
  
  print(model)
  print(f"Model is on cuda: {next(model.parameters()).is_cuda}")
  summary(model,input_size=x.size(),col_names=["input_size", "output_size", "num_params","params_percent","mult_adds","trainable"],col_width=15)

  #loss_fn = nn.BCEWithLogitsLoss()
  if useNLL: loss_fn = nn.NLLLoss(reduction='none') #expects log prob
  else: loss_fn = nn.CrossEntropyLoss(reduction='none') #expects logits
  if useBCE: loss_fn = nn.BCEWithLogitsLoss(reduction='none') #expects logits
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.1, weight_decay=1e-3)
  #optimizer = torch.optim.Adam(model.parameters(), lr=1)

  #output name
  outname="model_HL"
  if args.network=="Constituent": outname="model_const"
  if args.network=="Transformer": outname="model_transformer"
  print(f"Saving PyTorch model to {outname}.pth")
  print("Took %.3e seconds to make model"%((time.time()-starttime)))

  #run cycle
  epochs = 100
  loss_test=[]
  loss_train=[]
  acc_test=[]
  best_loss=1e6
  patience_counter=0
  patience=2
  for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    starttime=time.time()
    train(train_dataloader, model, loss_fn, optimizer, loss_train)
    print("Took %.2f minutes to run"%((time.time()-starttime)/60))
    starttime=time.time()
    test(test_dataloader, model, loss_fn, loss_test, acc_test)
    print("Took %.2f minutes to run"%((time.time()-starttime)/60))

    if loss_test[-1]<best_loss-0.01:
      best_loss=loss_test[-1]
      patience_counter=0
      torch.save(model.state_dict(), outname+".pth")
    else:
      patience_counter+=1
      if patience_counter>=patience:
        print("Early stoping")
        break
  print("Done!")

  #Some quick validation
  '''
  with torch.no_grad():
    for i in range(10):
      idx = torch.randint(len(test_data), size=(1,)).item()
      x, y = test_data[idx][0], test_data[idx][1]
      model.to("cpu")
      x=x.unsqueeze(0)
      pred=model(x)
      if useNLL: pred=torch.exp(model(x))
      else: pred=torch.sigmoid(model(x))
      if useBCE: 
        pred=torch.sigmoid(model(x).flatten())
        print(x.numpy(),pred.numpy(),pred.round().item(),y.numpy())
      else:
        print(x.numpy(),pred.numpy(),torch.argmax(pred).item(),y.numpy())
  '''

  #make some quick plots
  figure = plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel('Loss')
  plt.plot(loss_train)
  plt.plot(loss_test)
  plt.legend(['Train', 'Test'], loc='lower right')
  #plt.figtext(0.5, 0.5, perf[0], wrap=True, horizontalalignment='center', fontsize=10)
  plt.savefig(f"loss_{outname}.pdf")
  plt.close(figure)
  plt.show()
