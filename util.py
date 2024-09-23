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
