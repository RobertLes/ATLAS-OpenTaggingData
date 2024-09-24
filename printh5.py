#!/usr/bin/env python

import h5py
import sys,os
import numpy as np
import matplotlib.pyplot as plt


def make_hist(valueslist,name):
  fig, ax = plt.subplots()

  minval=min(np.min(valueslist[0]),np.min(valueslist[1]))
  maxval=max(np.max(valueslist[0]),np.max(valueslist[1]))
  #print(np.histogram_bin_edges(valueslist[0]))
  bins=np.linspace(minval,maxval,20)

  for values in valueslist:
    ax.hist(values,bins=bins,histtype="step")
  #plt.show()
  fig.savefig(name)
  plt.close(fig)

if __name__ == "__main__":
  if len(sys.argv)<2:
    print("No input files")
    sys.exit()

  infile=sys.argv[1]
  f = h5py.File(infile, 'r')

  #Make some printout
  print(f)
  for key,val in f.attrs.items():
    print("Attribute %s: %s" % (key, val))
  for name in f.keys():
    print(f[name])
    for key,val in (f[name]).attrs.items():
        print("    Atrribute %s: %s" % (key, val))

  if not os.path.exists("./Plots"): 
    os.mkdir("./Plots")

  #total number of jets, should be global
  Njet=f["labels"].len()
  Njet=10000

  #get lables
  labels=f["labels"][:Njet]

  # loop on collections
  for var in f.keys():
    print(var)
    #if f[var].ndim>1: continue
    #values=f[var][:Njet]
    values_sig=[]
    values_bkg=[]
    for index,label in enumerate(labels):
      if label==1:
        values_sig.append(f[var][index])
      if label==0:
        values_bkg.append(f[var][index])
    values_sig=np.asarray(values_sig).flatten()
    values_bkg=np.asarray(values_bkg).flatten()
    make_hist([values_sig,values_bkg],"Plots/plot_"+var+".pdf")
