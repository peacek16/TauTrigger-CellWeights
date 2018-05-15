import ROOT
from ROOT import TH1F, TCanvas, TFile, TColor,THStack,TLegend,TLatex, TMultiGraph, TVector3
from matplotlib import colors, pyplot
import numpy as np
from numpy import math
from mpl_toolkits.axes_grid1 import AxesGrid
ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)



"""
    File ouputs: Weight maps of epoch with acc < 0.001, acc plot, and resolution distribution
    Changed varaibles: string name of dataset, and acc limits 
    Environment setting: need a flat file output from the fNNs_simple.py   
"""

dataset = open("L2EtTau_ViEt15Et5Eta1.4_W.data","r")

ls_acc = []
ls_rms = []
sum_square = 0
sum_single = 0

hist = TH1F("Resolution","",100,-0.5,0.5)
hist.SetLineColor(ROOT.kBlue)
hist.SetLineWidth(3)

ls_run = []
run =0
canvas = {}
for line in dataset:
    run+=1
    ls_line = line.split(",")
    acc = float(ls_line[-1])    # acc = net-true/true
    ls_acc.append(acc)
    ls_run.append(run)
    hist.Fill(acc)
    sum_single+= acc      
    sum_square += ((acc)**2)
    if math.fabs(float(ls_line[-1])) < 0.001:     # optional; only print epoch with net-true/true < 0.001
         
        ls_L2Etcells = []
        arr_L2Etcells = np.asarray(ls_line[:-1])
        for i in range(85):
             ls_L2Etcells.append(float(ls_line[i]))
        arr_L2Etcells = np.asarray(ls_L2Etcells)
        extent = [0,17,5,0]
        arr_L2Etcells_2D = arr_L2Etcells.reshape((17,5)).transpose()
        
        fig, ax = pyplot.subplots()
        pyplot.title("L2Et 17x5 Weights: Adam Rate = 1, Epoch = "+str(run))
        im = ax.imshow(arr_L2Etcells_2D,vmin=-2, vmax=5,cmap='RdBu',extent=extent)
        pyplot.colorbar(im)
        pyplot.savefig("L2Et_EvalWeight_Run"+str(run)+".pdf")
           
dataset.close()

canvas["EvalAcc"] = ROOT.TCanvas()
ar_acc = np.array(ls_acc, dtype=np.float)
ar_run = np.array(ls_run, dtype=np.float)
g = ROOT.TGraph(len(ls_run),np.array(ar_run),np.array(ar_acc))
g.SetTitle("L2Et 17x5 : Adam Rate = 1, Epoch = 300 and Batch size = 500")
g.GetXaxis().SetTitle("epoch")
g.GetYaxis().SetTitle("net-true/true")
g.SetMarkerColor(ROOT.kRed)
g.SetMarkerSize(5)
g.Draw("AL")

canvas["EvalAcc"].SaveAs("L2Et_Weights_EvalAcc.pdf")

canvas["EvalResolution"] = ROOT.TCanvas()
leg = ROOT.TLegend(0.7,0.7,0.9,0.9)
leg.AddEntry(hist,"RMS = "+str(round(((sum_square/run) **0.5),2))+", Mean = "+str(round((sum_single/run),2)),"l")
hist.Draw()
leg.Draw()
hist.GetXaxis().SetTitle("(net-true)/true")
hist.GetYaxis().SetTitle("the number of seeds")
hist.SetTitle("L2Et 17x5 : Adam Rate = 1, Epoch = 300 and Batch size = 500") 
canvas["EvalResolution"].SaveAs("L2Et_EvalResolution.pdf")

