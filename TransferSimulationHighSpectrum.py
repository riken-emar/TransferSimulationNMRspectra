# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:39:37 2022

@author: Koki.H
"""

##########################
# Import libraries       #
##########################

import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
import datetime
import os
import scipy.signal as sig
import scipy.stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.cm as cm
import math
import csv

##########################
# Read tabletop NMRdata  #
##########################

fileName='130' 
dic, dataHighMagnet = ng.fileio.bruker.read(dir=r'C:\  "NMR TD file you want to read."  \\')
#Example
#dic, data = ng.fileio.bruker.read(dir=r'C:\**\**\**\**\\')

#Remove TOPSPIN digital filter
data = ng.bruker.remove_digital_filter(dic, dataHighMagnet)

#Make read acqus into a form of FID
BF1=dic['acqus']['BF1']
print('BF1',BF1)
O1=dic['acqus']['O1']
print("O1",O1)
TD=dic['acqus']['TD']/2
print("TD",TD)
y=data
O1p=dic['acqus']['O1']/dic['acqus']['BF1']
print("O1p",O1p)
ppmmin=math.floor(int(O1p-(dic['acqus']['SW']/2)))
print("ppmin",ppmmin)
ppmmax=math.floor(O1p+(dic['acqus']['SW']/2))
print("ppmax",ppmmax)
x=np.linspace(ppmmin,ppmmax,len(y))
at=(dic['acqus']['TD'])/dic['acqus']['SW_h']
print('at\n',at)
timeAxis = np.arange(0, at, at/TD)
FreqAxis = np.arange(ppmmin, ppmmax, dic['acqus']['SW']/dic['acqus']['TD'])
print('FreqAxis\n',FreqAxis)
fs=TD/at
print("fs\n",fs)
calib = 0
fig = plt.figure()
plt.plot(data)
now = datetime.datetime.now()
plt.title("1.FT Spectrum")
dirname = "HighTransferSimulationSpectra/"
os.makedirs(dirname, exist_ok=True)
filename = dirname + "FID" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

#############################
# Perform Fourier transform #
#############################
fig = plt.figure()
dataFT = ng.process.proc_base.fft(data)
plt.plot(dataFT)
plt.title("2.FT Spectrum")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "FT" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

################
# ↓ AutoPhase ↓#
################

fig = plt.figure()
dataAP = ng.process.proc_base.ps(dataFT)
plt.plot(dataAP)
plt.title("3.AutoPhase")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "AutoPhaseCollect" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

##########################
# Baseline correction    #
##########################

fig = plt.figure()
dataBL = ng.process.proc_bl.baseline_corrector(dataAP, wd=30)
plt.plot(dataBL)
print(dataBL)
plt.title("4.CorrectedBaseline")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "BaselineCollect" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

#############################    
# Inverse Fourier Transform #
#############################

fig = plt.figure()
dataIFFT = ng.process.proc_base.ifft(dataBL)
plt.plot(dataIFFT)
plt.title("5.IFFT")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "IFFT" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)


###############################################################
# Multiply window function to broaden high field NMR spectrum #
###############################################################

fig = plt.figure()
dataWindowFunc = ng.process.proc_base.em(dataIFFT,lb=0.0020960000229376006,inv=False,rev=False)
plt.plot(dataWindowFunc)
plt.title("6.dataTransfer")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "TransferFID" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

######################################################
#Define function to be used for fitting for frequency#
######################################################

fig = plt.figure()
#STFT
f,t, Zxxt = sig.stft(dataWindowFunc, 
                     fs=TD/at,
                     window='hamming',
                     nperseg=1024, 
                     noverlap=512,
                     return_onesided=False,
                     boundary=None,                   
                     padded=True,
                     axis = -1) 
X=np.fft.fftshift((1+np.abs(Zxxt)),axes=0)


print('f\n',f)
#fig=plt.figure()
#plt.plot(f)
print('t\n',t)
#fig=plt.figure()
#plt.plot(t)
print('Zxxt\n',Zxxt)
#fig=plt.figure()
#plt.plot(Zxxt)
print("X\n", X)
plt.plot(X)
fig = plt.figure()
ax=fig.add_subplot(111,projection="3d")
for i in range(X.shape[0]):
    l=[i]*X[i].shape[0]
    m=range(X[i].shape[0])
    n=X[i]
    #ax = fig.add_subplot(111, projection="3d")
    #ax.plot_surface(m,l,f,alpha=0.5)
    #plt.show()
    plt.plot(l,m,n,marker=".",alpha=0.5)
plt.tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False,labelsize=20)
ax.axis("off")
calib=256
fig = plt.figure()
X0=np.fft.fftshift(X,axes=0)
X0=np.fft.fftshift(X0,axes=0)
plt.pcolormesh(t, np.fft.fftshift(((f+O1+calib)/BF1)),X0,cmap='viridis')
plt.tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False,labelsize=20)
pp=plt.colorbar(orientation="vertical")
pp.ax.tick_params(labelsize=20)
plt.ylabel('$^{1}$H Chemical Shift [ppm]', fontsize=20)
plt.xlabel('Time [s]', fontsize=20)
plt.show()

#################################################################
#Matrix swap (time and frequency, to get frequency information) #
#################################################################

dataFreq = X.T
print("f\n", f)
plt.plot(dataFreq[0])
Frequency = dataFreq[0]

fig = plt.figure()
dataA = Frequency[0:500]
dataB = Frequency[600:1024]
dataC = np.append(dataA, dataB)
plt.plot(dataC)
plt.title("dataC")
plt.figure()

dataD = scipy.stats.zscore(dataC)
plt.plot(dataD)
plt.title("dataD")
plt.figure()

ave = np.mean(dataC)
std = np.std(dataC)
dataE = (Frequency - ave)/std
plt.plot(dataE)
plt.title("dataE")
plt.figure()

x=dataE
peaks, _ = find_peaks(x,height=1,distance=1)
plt.plot(x)
print("x\n", x)
plt.plot(peaks, x[peaks], "x")
print("PeakPoint\n", x[peaks])
print("peaks\n", peaks)
plt.show()




with open ("DataE0222.csv",'a',newline='')as f:
    writer = csv.writer(f)
    writer.writerow(dataE)

x1 = np.count_nonzero(dataFreq)
x2 = np.arange(x1)
print("x2\n",x2)
plt.figure()

xF = dataFreq[1]

with open ("MixtureParameterFreq.csv",'a',newline='')as f:
    
    writer = csv.writer(f)
    writer.writerow([fileName])
    writer.writerow(['Freq-Intensity','peakpoint','ChemicalShift','Freq-width'])
#    Mylist = ["Sample_name","Peak_number","Chemical_shift","Intensity","Frequency_width","T2_relaxation_Time"],

    
    #for i in range(len(peaks)): 
     #       writer.writerow([SampleName,peaks[i],ctr[i],amp[i],wid[i], T2[i]])
    def funcFreq(x, *params):
        num_func = int(len(params)/3)
        y_list = []
        for i in range(num_func):
            y = np.zeros_like(x)
            param_range = list(range(3*i,3*(i+1),1))
            amp = params[int(param_range[0])]
            #writer.writerow(amp)
            #print("強度", amp)
            ctr = params[int(param_range[1])]
            wid = params[int(param_range[2])]
            y = y + amp * np.exp( -((x - ctr)/wid)**2)
            y_list.append(y)
            y_sum = np.zeros_like(x)
            for i in y_list:
                y_sum = y_sum + i
            y_sum = y_sum + params[-1]
    
        return y_sum    
    
    def fit_plotFreq(x, *params):
        num_func = int(len(params)/3)
        y_list = []
        for i in range(num_func):
            y = np.zeros_like(x)


            chemicalShift=peaks*((ppmmax-ppmmin)/TD) +ppmmin   
            
            param_range = list(range(3*i,3*(i+1),1))
            amp = params[int(param_range[0])]
         #   writer.writerow([amp])
            ctr = params[int(param_range[1])]
          #  writer.writerow([ctr])
            wid = params[int(param_range[2])]
            print("幅\n",wid)
          #  writer.writerow([wid])
            writer.writerow([amp,peaks[i],chemicalShift[i],wid])
         #   writer.writerow()
            
            y = y + amp * np.exp( -((x - ctr)/wid)**2) + params[-1]
            y_list.append(y)
        return y_list
    fig = plt.figure()   
    
    guess = []
    for j in range(len(peaks)):
        guess.append([10000, peaks[j], 0.000005])
        print("peaks",peaks[j])  
        #writer.writerow([peaks])
        background = 0
        
    guess_total = []
    for i in guess:
        guess_total.extend(i)
    guess_total.append(background)
    
    x = np.arange(len(Frequency))
    popt, pcov = curve_fit(funcFreq, x, Frequency, p0=guess_total)
    print('x\n',x)
    print('Frequency\n',Frequency)
    print('LenFrequency\n',len(Frequency))
    

    
    fit = funcFreq(x, *popt)
    print('Fit',len(fit))
    #FreqAxis = np.arange(ppmmin, ppmmax, dic['acqus']['SW']/len(Frequency))
    FreqAxis=np.linspace(ppmmin,ppmmax,len(fit))
    axes = plt.axes()
    axes.set_xlim([ppmmax,ppmmin])
    plt.scatter(FreqAxis, Frequency, s=20)
    plt.plot(FreqAxis, fit , ls='-', c='black', lw=1)

    y_list = fit_plotFreq(FreqAxis, *popt)
    baseline = np.zeros_like(x) + popt[-1]
    for n,i in enumerate(y_list):
        plt.fill_between(FreqAxis, i, baseline, facecolor=cm.rainbow(n/len(y_list)), alpha=0.6)
    plt.title("Freq Fitting") 
       
with open ("MixtureParameterTime.csv",'a',newline='')as f:
    
    writer = csv.writer(f)
    writer.writerow([fileName])
    writer.writerow(['Time-Intensity','T2RelaxationTime'])



    def funcTime(Time, *params):        
        num_func = int(len(params)/3)
        y_list = []
        for i in range(num_func):
            y = np.zeros_like(Time)
            param_range = list(range(3*i,3*(i+1),1))
            a = params[int(param_range[0])]
            #print(a)
            W = params[int(param_range[1])]
            #print(W)
            T2 = params[int(param_range[2])]
            #print(T2)
            y = y + a*np.exp(-1 *(Time/T2)**W)
            y_list.append(y)
        y_sum = np.zeros_like(Time)
        for i in y_list:
            y_sum = y_sum + i     
            
        y_sum = y_sum + params[-1]
    
        return y_sum            
            
    def fit_plotTime(Time, *params):
        num_func = int(len(params)/3)
        y_list = []
        for i in range(num_func):
            y = np.zeros_like(Time)
            param_range = list(range(3*i,3*(i+1),1))
    
            a = params[int(param_range[0])]
           # print("強度\n",a)
            W = params[int(param_range[1])]
            print("Weibull",W)
            T2 = params[int(param_range[2])]
            print("T2\n",T2*at/len(x))
            y = y + a*np.exp(-1/W *(Time/T2)**W)
            y_list.append(y)
            writer.writerow([a,T2/fs])
            print("Time\n",Time*at/len(x))
            #print(np.max(y_list[i]))
    
        return y_list            
            
    guess = []
    guess.append([1, 2, 1])            
    guess_total = []
    for i in guess:
        guess_total.extend(i)
    guess_total.append(background)
                
    param_bounds=([1,1,0.001,0],[np.inf,2,np.inf,10])            
            
    for j in range(len(peaks)):
        slice=X[peaks[j],:]
        fig=plt.figure()
        plt.plot(slice)
        print("sliceNo.\n",)
        y = slice
        print(slice)
        x1 = np.count_nonzero(y)
        x = np.arange(x1)
        
        print(x)
        print(y)
        print(guess_total)
        popt, pcov = curve_fit(funcTime, x, y, p0=guess_total, bounds=param_bounds)

        print(popt)
        fig=plt.figure()
        fit = funcTime(x, *popt)
        plt.scatter(x*at/len(x), y, s=20, c='black')
        plt.plot(x*at/len(x), fit , ls='-', c='black', lw=1)
        

    
        y_list = fit_plotTime(x, *popt)
        baseline = np.zeros_like(x) + popt[-1]
       # chemicalShift=peaks*((ppmmax-ppmmin)/TD) +ppmmin 
        for n,i in enumerate(y_list):
            plt.fill_between(x*at/len(x), i , baseline, facecolor=cm.rainbow(n/len(y_list)), alpha=0.6)
        plt.show()
            
            
            
            
            
            
            
            