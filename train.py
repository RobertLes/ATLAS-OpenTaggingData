#!/usr/bin/env python
# coding: utf-8

from utils import *
import torch
from weaver.nn.model.ParticleTransformer import ParticleTransformer

import matplotlib.pyplot as plt
import sys,os,time
import argparse

def train(dataloader, model, loss_fn, optimizer, loss_train):
  #set to train mode
  model.train()

  #Loop over batches
  size = len(dataloader.dataset)
  for batch, (X, y, w) in enumerate(dataloader):
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

  #Get input agrs
  parser = argparse.ArgumentParser()
  parser.add_argument("-n","--network", help="The type of network to run",choices=['HL', 'Constituent', 'Transformer'],required=True)
  parser.add_argument("-d","--directory", default="./", help="The directory with the input files")
  args = parser.parse_args()

  #Get dataloaders, function in utils.py
  starttime=time.time()
  train_dataloader,test_dataloader=get_ATLAS_inputs(args.directory,args.network,batch_size=2**8)
  print("Took %.3e seconds to run data loading\n"%((time.time()-starttime)))

  #Quick debug on input shape
  for x, y, z in train_dataloader:
    print(f"Shape of x: {x.shape}, {x.dtype}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    print(f"Shape of z: {z.shape} {z.dtype}")
    #print(x,y,z)
    break

  # Get cpu, gpu or mps device for training.
  starttime=time.time()
  device = ( "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
  if device=="cuda":
    print(f"\nUsing device {device,torch.cuda.get_device_name(0)}\n")
  else:
    print(f"\nUsing device \"{device}\"\n")

  # Define model
  if args.network=="Constituent": Ndim = torch.flatten(x[0]).size(dim=0)
  elif args.network=="Transformer": Ndim=x.size(dim=1)
  else: Ndim = x.size(dim=1)
  if args.network=="Transformer": model = ParticleTransformer(input_dim=Ndim,num_classes=2).to(device)
  else: model = DNNetwork(Ndim,useConstituents=(args.network=="Constituent")).to(device)

  '''
  if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    #model = nn.parallel.DistributedDataParallel(model)
  '''
  
  print(model)
  #print(f"Model is on cuda?: {next(model.parameters()).is_cuda}")
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
  print("\nTook %.3e seconds to make model"%((time.time()-starttime)))
  print(f"Will save PyTorch model to {outname}.pth")

  #run cycle
  epochs = 100
  loss_test=[]
  loss_train=[]
  acc_test=[]
  best_loss=1e6
  patience_counter=0
  patience=5
  for t in range(epochs):
    print(f"\nEpoch {t+1}\n-------------------------------")
    starttime=time.time()
    train(train_dataloader, model, loss_fn, optimizer, loss_train)
    print("Took %.2f minutes to run"%((time.time()-starttime)/60))
    starttime=time.time()
    test(test_dataloader, model, loss_fn, loss_test, acc_test)
    print("Took %.2f minutes to run"%((time.time()-starttime)/60))

    if loss_test[-1]<best_loss:
      best_loss=loss_test[-1]
      patience_counter=0
      print(f"Saving model since best loss")
      torch.save(model.state_dict(), outname+".pth")
    else:
      patience_counter+=1
      if patience_counter>=patience:
        print("Early stoping")
        break
  print("Done!")

  #make some quick plots
  figure = plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel('Loss')
  plt.plot(loss_train)
  plt.plot(loss_test)
  plt.legend(['Train', 'Test'], loc='lower right')
  plt.savefig(f"loss_{outname}.pdf")
  plt.close(figure)
  plt.show()
