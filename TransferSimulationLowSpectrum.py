# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 19:22:41 2022

@author: Koki.H
"""

##########################
# Import a library       #
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
import csv


##########################
# Read tabletop NMRdata  #
##########################

dic2, data2 = ng.fileio.jcampdx.read(r'  "NMR TD file you want to read."  ')
# Example
# dic, data = ng.fileio.jcampdx.read(r'C:\**\**\**\**\20220222.td')

print("RealData\n", data2[0])
print("ImaginaryData\n", data2[1])

#Specify the name of the directory to be saved.
dirname = "LowSpectrumFolder/"

###########################################
# Adding complex numbers and real numbers #
###########################################

fig = plt.figure()
dataFID = data2[0]+(data2[1]*1j)
plt.plot(dataFID)
plt.title("1.FID")
now = datetime.datetime.now()
# Save FID
os.makedirs(dirname, exist_ok=True)
filename = dirname + "FID" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

##########################
# Zero　filling 　　       #
##########################
fig = plt.figure()

pointAdjust=15
pa=pointAdjust
dataZeroFilling = ng.process.proc_base.zf(dataFID,pad=1024*pa,mid=False)
plt.plot(dataZeroFilling)
plt.title("2.ZeroFilling")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "IFFT" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

dataPointNum = 1024+1024*pa
fs = dataPointNum
print('fs\n',fs)

TD = len(dataZeroFilling)
print('TD\n',TD)
SWH = 12
at = TD/SWH

#############################
# Perform Fourier transform #
#############################

fig = plt.figure()
dataFT = ng.process.proc_base.fft(dataZeroFilling)
plt.plot(dataFT)
plt.title("3.FT Spectrum")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "FT" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

###########################
# Manual Phase correction #
###########################

fig = plt.figure()
p0, p1 = ng.process.proc_autophase.manual_ps(dataFT)
dataManualPhaseCollect = ng.proc_base.ps(dataFT, p0=p0, p1=p1)
os.makedirs(dirname, exist_ok=True)
filename = dirname + "ManualPhaseCollect" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

#################################
# ManualPhase reflected in Auto #
#################################

fig = plt.figure()
dataAutoPhase = ng.process.proc_base.ps(dataManualPhaseCollect)
plt.plot(dataAutoPhase)
plt.title("4.PhaseCollect")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "AutoPhaseCollect" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

##########################
# Baseline correction    #
##########################

fig = plt.figure()
dataBaseLineCollect = ng.process.proc_bl.baseline_corrector(dataManualPhaseCollect, wd=20)
plt.plot(dataBaseLineCollect)
plt.title("5.BaseLineCollect")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "BaselineCollect" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

#############################
# Inverse Fourier transform #
#############################

fig = plt.figure()
dataIFFT = ng.process.proc_base.ifft(dataBaseLineCollect)
plt.plot(dataIFFT)
plt.title("6.IFFT")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "IFFT" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

###########################################
# Perform standard deviation calculations #
###########################################

fig = plt.figure()
dataLeft = dataBaseLineCollect[0*pa:460*pa]
dataRight = dataBaseLineCollect[610*pa:1000*pa]
dataZF=dataBaseLineCollect[460*pa+1:610*pa-1]*0
dataRL = np.append(dataLeft, dataRight)
dataWaterZero2 = np.append(dataLeft, dataZF)
dataWaterZero = np.append(dataWaterZero2, dataRight)
plt.plot(dataRL)
plt.title("dataRL")
plt.figure()

dataZScore = scipy.stats.zscore(dataRL)
plt.plot(dataZScore)
plt.title("dataZScore")
plt.figure()

ave = np.mean(dataRL)
std = np.std(dataZScore)
dataNormalize = (dataBaseLineCollect- ave)/std
plt.plot(dataNormalize)
plt.title("dataNormalize")
plt.figure()

peaks, _ = find_peaks(dataNormalize,height=900,distance=1)
plt.plot(dataNormalize)
print("x\n", dataNormalize)
plt.plot(peaks, dataNormalize[peaks], "x")
print("PeakPoint\n", dataNormalize[peaks])
print("peaks\n", peaks)
plt.show()

###################################
# Inverse Fourier transform of RL #
###################################

fig = plt.figure()
dataIFFT2 = ng.process.proc_base.ifft(dataWaterZero)
plt.plot(dataIFFT2)
plt.title("6.IFFT2")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "IFFT" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

#############################
# Multiply window functions #
#############################

fig = plt.figure()
dataWindowFunc = ng.process.proc_base.em(dataIFFT2,lb=0.001,inv=False,rev=False)
plt.plot(dataWindowFunc)
plt.title("7.WindowFunction")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "WindowFuncFID" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

##############################
# STFT to generate 3D matrix #
##############################

fig = plt.figure()
f, t, Zxxt = sig.stft(dataWindowFunc, fs=1024/dataPointNum, 
                  window='hamming', 
                  nperseg=256, 
                  noverlap=128, 
                  nfft=None, 
                  detrend=False, 
                  return_onesided=False, 
                  boundary=None, 
                  padded=True, 
                  axis=- 1)
STFTMatrix = np.fft.fftshift((np.abs(Zxxt)), axes=0)
print("X\n", STFTMatrix)
plt.plot(STFTMatrix)
fig = plt.figure()
ax=fig.add_subplot(111,projection="3d")
for i in range(STFTMatrix.shape[0]):
    l=[i]*STFTMatrix[i].shape[0]
    m=range(STFTMatrix[i].shape[0])
    n=STFTMatrix[i]
    #ax = fig.add_subplot(111, projection="3d")
    #ax.plot_surface(m,l,f,alpha=0.5)
    #plt.show()
    plt.plot(l,m,n,marker=".",alpha=0.5)
plt.tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False,labelsize=20)
ax.axis("off")
plt.figure()



#plt.show() 



dataFreq = STFTMatrix.T
print("f\n", f)
plt.plot(dataFreq[0])
plt.figure()

Frequency = dataFreq[0]

dataSTFTLeft = Frequency[0:200]
dataSTFTRight = Frequency[240:512]
dataSTFTRL = np.append(dataSTFTLeft, dataSTFTRight)
plt.plot(dataSTFTRL)
plt.title("dataSTFTRL")
plt.figure()

dataSTFTZscore = scipy.stats.zscore(dataSTFTRL)
plt.plot(dataSTFTZscore)
plt.title("dataSTFTZscore")
plt.figure()

ave = np.mean(dataSTFTRL)
std = np.std(dataSTFTRL)
dataE = (Frequency- ave)/std
plt.plot(dataE)
plt.title("dataE")
plt.figure()

x=dataE
peaks, _ = find_peaks(x,height=0.1,distance=1)
plt.plot(x)
print("x\n", x)
plt.plot(peaks, x[peaks], "x")
print("PeakPoint\n", x[peaks])
print("peaks\n", peaks)
plt.show()

x1 = np.count_nonzero(dataFreq)
x2 = np.arange(x1)
print("x2\n",x2)
plt.figure()

#####################################
# Frequency directional fit of STFT #
#####################################

def funcFreq(Frequency, *params):
    #params length determines the number of functions to be fitted.
    num_func = int(len(params)/3)
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(Frequency)
        param_range = list(range(3*i,3*(i+1),1))
        amp = params[int(param_range[0])]
        #print("強度", amp)
        ctr = params[int(param_range[1])]
        wid = params[int(param_range[2])]
        print("幅\n",wid)
        y = y + amp * np.exp( -((Frequency - ctr)/wid)**2)
        y_list.append(y)
    y_sum = np.zeros_like(Frequency)
    for i in y_list:
        y_sum = y_sum + i
    
    y_sum = y_sum + params[-1]
    return y_sum 
    
def fit_plotFreq(Frequency, *params):
    num_func = int(len(params)/3)
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(Frequency)
        param_range = list(range(3*i,3*(i+1),1))
        amp = params[int(param_range[0])]
        ctr = params[int(param_range[1])]
        wid = params[int(param_range[2])]
        y = y + amp * np.exp( -((Frequency - ctr)/wid)**2) + params[-1]
        y_list.append(y)
    return y_list
fig = plt.figure()   
guess = []
for j in range(len(peaks)):
    guess.append([10000, peaks[j], 0.1])
    print("peaks",peaks[j])    
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
with open("TransferSimulationLowSpectrum","a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow(Frequency)

fit = funcFreq(x, *popt)
print('Fit',len(fit))
#FreqAxis = np.arange(ppmmin, ppmmax, dic['acqus']['SW']/len(Frequency))
FreqAxis=np.linspace(0,12,len(fit))    
    
plt.scatter(FreqAxis, Frequency, s=20)
plt.plot(FreqAxis, fit , ls='-', c='black', lw=1)

y_list = fit_plotFreq(FreqAxis, *popt)
baseline = np.zeros_like(x) + popt[-1]
for n,i in enumerate(y_list):
    plt.fill_between(FreqAxis, i, baseline, facecolor=cm.rainbow(n/len(y_list)), alpha=0.6)
plt.title("Freq Fitting")

now = datetime.datetime.now()
dirname = "FreqFitting_Low/"
os.makedirs(dirname, exist_ok=True)
filename = dirname + "FT" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

def funcTime(x, *params):
    num_func = int(len(params)/3)

    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        a = params[int(param_range[0])]
        #print(a)
        W = params[int(param_range[1])]
        #print(W)
        T2 = params[int(param_range[2])]
        #print(T2)
        y = y + a*np.exp(-1/W *(x/T2)**W)
        y_list.append(y)
    y_sum = np.zeros_like(x)
    for i in y_list:
        y_sum = y_sum + i
        
    y_sum = y_sum + params[-1]

    return y_sum

def fit_plotTime(x, *params):
    num_func = int(len(params)/3)
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        a = params[int(param_range[0])]
       # print("強度\n",a)
        W = params[int(param_range[1])]
        #print(W)
        T2 = params[int(param_range[2])]
        print("T2\n",T2*at/len(x))
        y = y + a*np.exp(-1/W *(x/T2)**W)
        y_list.append(y)
        #print(np.max(y_list[i]))

    return y_list

guess = []
guess.append([1, 2, 1])

background = 0
guess_total = []
for i in guess:
    guess_total.extend(i)
guess_total.append(background)
param_bounds=([1,1,0.0000001,0],[np.inf,2,np.inf,10])
for j in range(len(peaks)):
    slice=STFTMatrix[peaks[j],:]
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


