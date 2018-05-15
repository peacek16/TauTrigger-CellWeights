import ROOT
from ROOT import TH1F, TCanvas, TFile, TColor,THStack,TLegend,TLatex, TMultiGraph, TVector3
from matplotlib import colors, pyplot
import numpy as np
from numpy import math
from mpl_toolkits.axes_grid1 import AxesGrid
ROOT.gROOT.SetBatch(1) #PK: display from remote ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)

"""
    File ouputs: Multi-Layer Weight maps of epoch with acc < 0.001, acc plot, and resolution distribution
    Changed varaibles: string name of dataset, and acc limits 
                       if don't need Had layer, change the string to "NULL" + decrease the number of loop by 1 in j + add another zero tensor 
    Environment setting: need a flat file output from the fNNs_simple.py   
"""

dataset = open("LEtTau_ViEt15Et5Eta1.4_W.data","r")

ls_acc = []
ls_rms = []
sum_square = 0
sum_single = 0

hist = TH1F("Resolution","",100,-0.5,0.5)
hist.SetLineColor(ROOT.kBlue)
hist.SetLineWidth(3)

ls_run = []
ls_title = ["EM Layer0 Et", "EM Layer1 Et", "EM Layer2 Et", "EM Layer3 Et", "Had Layer Et", "NULL"]
run =0
canvas = {}
for line in dataset:
    arr_LEt = {}
    ls_LEt = []
    run+=1
    ls_L2Etcells = []
    ls_line = line.split(",")
    acc = float(ls_line[-1])    # acc = net-true/true 
    ls_acc.append(acc)
    ls_run.append(run)
    hist.Fill(acc)
    sum_single+= acc
    sum_square += ((acc)**2)
    if math.fabs(float(ls_line[-1])) < 0.001:       # optional; only print epoch with net-true/true < 0.001
        ls_LEtcells = []
        for i in range(85*5):
             ls_LEtcells.append(float(ls_line[i])) 
        arr_LEtcells = np.asarray(ls_LEtcells)
        extent = [0,17,5,0]
        arr_LEtcells_3D = arr_LEtcells.reshape((5,17,5))

        
        fig, ax = pyplot.subplots()
        pyplot.title("LEt 17x5 Weights: Adam Rate = 1, Epoch = "+str(run))
        pyplot.axis('off')
        im = ax.imshow(arr_L2Etcells_2D,vmin=-2, vmax=5,cmap='RdBu',extent=extent)
        pyplot.colorbar(im)
        
        
        for j in range(0,5):
            arr_LEt[j] = arr_LEtcells_3D[j].transpose()
            ls_LEt.append(arr_LEt[j])
          
        ls_LEt.append(np.asarray([0]*85).reshape(17,5).transpose())    # for "NULL" add more if delete Had layer 

        grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 3),
                axes_pad=0.2,
                share_all=True,
                label_mode="L",
                cbar_location="right",
                cbar_mode="single",
                )
        c =0
        for val, ax in zip(ls_LEt,grid):
            c+=1
            im = ax.imshow(val, vmin=-2, vmax=5, cmap='RdBu',extent=extent)
            ax.set_xticks(np.arange(0,18,1))
            ax.set_yticks(np.arange(0,6,1))
            ax.set_title(ls_title[c-1], fontsize=7)
            ax.xaxis.set_tick_params(labelsize=4)
            ax.yaxis.set_tick_params(labelsize=4)
        grid.cbar_axes[0].colorbar(im)

        for cax in grid.cbar_axes:
            cax.toggle_label(True)
        
        pyplot.savefig("LEt_EvalWeight_Run"+str(run)+".pdf")
  
      
 
      
dataset.close()

canvas["EvalAcc"] = ROOT.TCanvas()
ar_acc = np.array(ls_acc, dtype=np.float)
ar_run = np.array(ls_run, dtype=np.float)
g = ROOT.TGraph(len(ls_run),np.array(ar_run),np.array(ar_acc))
g.SetTitle("LEt 17x5 : Adam Rate = 1, Epoch = 300 and Batch size = 500")
g.GetXaxis().SetTitle("epoch")
g.GetYaxis().SetTitle("net-true/true")
g.SetMarkerColor(ROOT.kRed)
g.SetMarkerSize(5)
g.Draw("AL")

canvas["EvalAcc"].SaveAs("LEt_Weights_EvalAcc.pdf")

canvas["EvalResolution"] = ROOT.TCanvas()
leg = ROOT.TLegend(0.7,0.7,0.9,0.9)
leg.AddEntry(hist,"RMS = "+str(round(((sum_square/run) **0.5),2))+", Mean = "+str(round((sum_single/run),2)),"l")
hist.Draw()
leg.Draw()
hist.GetXaxis().SetTitle("(net-true)/true")
hist.GetYaxis().SetTitle("the number of seeds")
hist.SetTitle("LEt 17x5 : Adam Rate = 1, Epoch = 300 and Batch size = 500") 
canvas["EvalResolution"].SaveAs("LEt_EvalResolution.pdf")
