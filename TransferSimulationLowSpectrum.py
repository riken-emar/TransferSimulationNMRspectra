# -*- coding: utf-8 -*-
"""
RIKEN EMAR 2022
@author: Koki Hara and Shunji Yamada
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
import matplotlib
import ntpath
import time

##########################
# Read tabletop NMRdata  #
##########################
#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160420_ninjinnama_d2o.td'
#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160419_ninjinji10min_d2o.td'
#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160419_ninjinji30m_d2o.td'
#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160419_ninjinni1h_d2o.td'

#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160226_yogurt0h_d2o.td'
#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160419_yogurt1.5h_d2o.td'
#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160420_yogurt3h_d2o.td'
#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160420_yogurt6h_d2o.td'
#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160420_yogurt8.5hh_d2o.td'
#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160420_yogurt20h_d2o.td'
#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160420_yogurt24h_d2o.td'
#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160420_yogurt48h_d2o.td'

filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160415_magurosashi-d0_d2o.td'
#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160226_maguro_sashi_d0_d2o.td'
#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160209_hirame_d2o.td'
#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160307_iwashisashi_d2o.td'
#filepath = r'C:\Bruker\TopSpin4.1.3\data\benchtop_NMR_dataset\20160415_iwashisashi_d2o.td'
jdxpath = filepath[:-3] + ".jdx"
dic1, data1 = ng.fileio.jcampdx.read(jdxpath)

now = datetime.datetime.now()
dirname = "benchtopNMRdata/" +now.strftime('%Y%m%d_%H%M%S') + "_" + ntpath.split(filepath)[1][:-3] + "/"

os.mkdir(dirname)
now = datetime.datetime.now()
datafilename = dirname + "output" +now.strftime('%Y%m%d_%H%M%S') + "_" + ntpath.split(filepath)[1][:-3] + ".csv"#["filename", ntpath.split(filepath)[1]]
#with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
with open(datafilename,"a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow([datafilename])

dic2, data2 = ng.fileio.jcampdx.read(filepath)

# Example
# dic, data = ng.fileio.jcampdx.read(r'C:\**\**\**\**\20220222.td')
#print("dic\n", dic2)
datadic = ["dic"]
#with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
with open(datafilename,"a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow(datadic)
    writer.writerow(dic1)
    writer.writerow(dic1.values())
    writer.writerow(dic2)
    writer.writerow(dic2.values())
#print("RealData\n", data2[0])
#print("ImaginaryData\n", data2[1])
RealData = ["RealData"]
ImaginaryData = ["ImaginaryData"]
#with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
with open(datafilename,"a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow(RealData)
    writer.writerow(data2[0])
    writer.writerow(ImaginaryData)
    writer.writerow(data2[1])
#Specify the name of the directory to be saved.
#dirname = "LowSpectrumFolder/"

###########################################
# Adding complex numbers and real numbers #
###########################################
at = float(dic2["LAST"][0][0:6])#1.4322
print('at\n',at)
fig = plt.figure()
dataFID = data2[0]+(data2[1]*1j)
TimeAxis=np.linspace(0,at,len(dataFID)) 
plt.plot(TimeAxis, dataFID)
plt.title("1.FID")
now = datetime.datetime.now()
# Save FID
os.makedirs(dirname, exist_ok=True)
filename = dirname + "FID" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
plt.xlabel("Time[sec]")
fig.savefig(filename)

plt.close('all')

FIDData = [filename]
#with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
with open(datafilename,"a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow(FIDData)
    writer.writerow(["dataFID.real"])
    writer.writerow(dataFID.real)
    writer.writerow(["dataFID.imag"])
    writer.writerow(dataFID.imag)

##########################
# Zero　filling 　　       #
##########################
fig = plt.figure()

pointAdjust=15
pa=pointAdjust
dataZeroFilling = ng.process.proc_base.zf(dataFID,pad=1024*pa,mid=False)
TimeAxis=np.linspace(0,at,len(dataZeroFilling)) 

plt.plot(TimeAxis, dataZeroFilling)
plt.title("2.Zero Filling")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "Zerofilling" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
plt.xlabel("Time[sec]")
fig.savefig(filename)

plt.close('all')

ZeroFilling = [filename]
#with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
with open(datafilename,"a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow(ZeroFilling)
    writer.writerow(["dataZeroFilling.real"])
    writer.writerow(dataZeroFilling.real)
    writer.writerow(["dataZeroFilling.imag"])
    writer.writerow(dataZeroFilling.imag)

dataPointNum = 1024+1024*pa
print('dataPointNum\n',dataPointNum)

PointNum = ["dataPointNum"]
#with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
with open(datafilename,"a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow(PointNum)
    writer.writerow([dataPointNum])

TD = len(dataZeroFilling)
#print('TD\n',TD)
SW = float(dic2["$SWEEPWIDTH"][0][0:3])#12
print('SW\n',SW)

SpectralWidth = ["SpectralWidth"]
AcquisitionTime = ["AcquisitionTime"]
#with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
with open(datafilename,"a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow(SpectralWidth)
    writer.writerow([SW])
    writer.writerow(AcquisitionTime)
    writer.writerow([at])

#############################
# Perform Fourier transform #
#############################

fig = plt.figure()
dataFT = ng.process.proc_base.fft(dataZeroFilling)
#print(dic1["FIRSTX"][0])
#print(dic1["LASTX"][0])
#print(dic2["$SFO1"][0])
ppmfirst = float(dic1["FIRSTX"][0])/float(dic2["$SFO1"][0])#-63.481316/60
ppmlast = float(dic1["LASTX"][0])/float(dic2["$SFO1"][0])#650.717223/60
#FreqAxis=np.linspace(0,12,len(fit))
FreqAxis=np.linspace(ppmfirst,ppmlast,len(dataFT)) 
plt.plot(FreqAxis,dataFT)
plt.title("3.FT Spectrum")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "FT" + now.strftime('%Y%m%d_%H%M%S') + "img.png"
plt.xlabel("ChemicalShift[ppm]")
plt.gca().invert_xaxis()
fig.savefig(filename)

plt.close('all')

FTSpectrum = [filename]
#with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
with open(datafilename,"a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow(FTSpectrum)
    writer.writerow(["dataFT.real"])
    writer.writerow(dataFT.real)
    writer.writerow(["dataFT.imag"])
    writer.writerow(dataFT.imag)

#################################
# Autophase Correction #
#################################

fig = plt.figure()
dataAutophase = ng.process.proc_base.ps(dataFT, p0=15, p1=0)
#FreqAxis=np.linspace(0,12,len(fit))
FreqAxis=np.linspace(ppmfirst,ppmlast,len(dataAutophase)) 

plt.plot(FreqAxis,dataAutophase)
plt.title("4.Autophase Correction")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "AutophaseCorrection" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
plt.xlabel("ChemicalShift[ppm]")
plt.gca().invert_xaxis()
fig.savefig(filename)

plt.close('all')

dataAutophasefilename = [filename]
#with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
with open(datafilename,"a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow(dataAutophasefilename)
    writer.writerow(["dataAutophase.real"])
    writer.writerow(dataAutophase.real)
    writer.writerow(["dataAutophase.imag"])
    writer.writerow(dataAutophase.imag)

###########################
# Manual Phase correction #
###########################
FreqAxispoint=range(len(dataAutophase)) 
fig = plt.figure()
plt.title("5.Manual Phase Correction")
plt.xticks([FreqAxispoint[-1],(FreqAxispoint[-1]/2),FreqAxispoint[0]],labels=[round(ppmlast,1),round((ppmlast+ppmfirst)/2,1),round(ppmfirst,1)])
plt.xlabel("ChemicalShift[ppm]")
plt.gca().invert_xaxis()
p0, p1 = ng.process.proc_autophase.manual_ps(dataAutophase)
dataManualPhaseCorrection = ng.proc_base.ps(dataAutophase, p0=p0, p1=p1)
os.makedirs(dirname, exist_ok=True)
filename = dirname + "ManualPhaseCorrection" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

plt.close('all')

dataManualPhaseCorrectionfilename = [filename]
#with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
with open(datafilename,"a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow(dataManualPhaseCorrectionfilename)
    writer.writerow(["p0"])
    writer.writerow([p0])
    writer.writerow(["p1"])
    writer.writerow([p1])
    writer.writerow(["dataManualPhaseCorrection.real"])
    writer.writerow(dataManualPhaseCorrection.real)
    writer.writerow(["dataManualPhaseCorrection.imag"])
    writer.writerow(dataManualPhaseCorrection.imag)

#################################
# Autophase Correction after manual #
#################################

fig = plt.figure()
dataAutophase = ng.process.proc_base.ps(dataManualPhaseCorrection, p0=15, p1=0)
#FreqAxis=np.linspace(0,12,len(fit))
FreqAxis=np.linspace(ppmfirst,ppmlast,len(dataAutophase)) 

plt.plot(FreqAxis,dataAutophase)
plt.title("6.AutophaseCorrection after ManualPhaseCorrection")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "AutophaseCorrectionAfterManualPhaseCorrection" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
plt.xlabel("ChemicalShift[ppm]")
plt.gca().invert_xaxis()
fig.savefig(filename)

plt.close('all')

dataAutoPhasefilename = [filename]
#with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
with open(datafilename,"a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow(dataAutoPhasefilename)
    writer.writerow(["dataAutophase.real"])
    writer.writerow(dataAutophase.real)
    writer.writerow(["dataAutophase.imag"])
    writer.writerow(dataAutophase.imag)


##########################
# Baseline correction    #
##########################

fig = plt.figure()
dataBaselineCorrection = ng.process.proc_bl.baseline_corrector(dataManualPhaseCorrection, wd=20)
FreqAxis=np.linspace(ppmfirst,ppmlast,len(dataBaselineCorrection)) 
plt.plot(FreqAxis,dataBaselineCorrection)
plt.title("7.Baseline Correction")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "BaselineCorrection" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
plt.xlabel("ChemicalShift[ppm]")
plt.gca().invert_xaxis()
fig.savefig(filename)

plt.close('all')

dataBaselineCorrectionfilename = [filename]
#with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
with open(datafilename,"a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow(dataBaselineCorrectionfilename)
    writer.writerow(["dataBaselineCorrection.real"])
    writer.writerow(dataBaselineCorrection.real)
    writer.writerow(["dataBaselineCorrection.imag"])
    writer.writerow(dataBaselineCorrection.imag)

#############################
# Inverse Fourier transform #
#############################

fig = plt.figure()
dataIFFT = ng.process.proc_base.ifft(dataBaselineCorrection)
TimeAxis=np.linspace(0,at,len(dataIFFT)) 
plt.plot(TimeAxis,dataIFFT)
plt.title("8.IFFT")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "IFFT" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
plt.xlabel("Time[sec]")
fig.savefig(filename)

plt.close('all')

dataIFFTfilename = [filename]
#with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
with open(datafilename,"a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow(dataIFFTfilename)
    writer.writerow(["dataIFFT.real"])
    writer.writerow(dataIFFT.real)
    writer.writerow(["dataIFFT.imag"])
    writer.writerow(dataIFFT.imag)

###########################################
# Perform standard deviation calculations #
###########################################


#dataLeft = dataBaselineCorrection[0*pa:460*pa]
#dataRight = dataBaselineCorrection[610*pa:1024*pa]
#dataZF=dataBaselineCorrection[460*pa+1:610*pa-1]*0
dataLeft = dataBaselineCorrection[0:7000]
dataRight = dataBaselineCorrection[8500:]
dataZF=dataBaselineCorrection[7000:8500]*0
#dataRL = np.append(dataLeft, dataRight)
dataWaterZero2 = np.append(dataLeft, dataZF)
dataWaterZero = np.append(dataWaterZero2, dataRight)
#plt.plot(dataRL)
#plt.title("dataRL")
#plt.figure()

#dataZScore = scipy.stats.zscore(dataRL)
dataZScore = scipy.stats.zscore(dataWaterZero)
fig = plt.figure()
plt.plot(FreqAxis,dataZScore)
plt.title("9.Delete water signal + ZScore")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "DeleteWater_ZScore" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
plt.xlabel("ChemicalShift[ppm]")
plt.gca().invert_xaxis()
fig.savefig(filename)

plt.close('all')

###################################
# Inverse Fourier transform of RL #
###################################

fig = plt.figure()
dataIFFT2 = ng.process.proc_base.ifft(dataWaterZero)
plt.plot(TimeAxis,dataIFFT2)
plt.title("10.IFFT after deleting water signal")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "IFFT_DeleteWater" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
plt.xlabel("Time[sec]")
fig.savefig(filename)

plt.close('all')

#############################
# Multiply window functions #
#############################

fig = plt.figure()
dataWindowFunc = ng.process.proc_base.em(dataIFFT2,lb=0.001,inv=False,rev=False)
plt.plot(TimeAxis,dataWindowFunc)
plt.title("11.Window Function")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "WindowFunction" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
plt.xlabel("Time[sec]")
fig.savefig(filename)

plt.close('all')

##############################
# STFT to generate 3D matrix #
##############################


f, t, Zxx = sig.stft(dataWindowFunc, fs=1024/dataPointNum, 
                  window='hamming', 
                  nperseg=256, 
                  noverlap=128, 
                  nfft=None, 
                  detrend=False, 
                  return_onesided=False, 
                  boundary=None, 
                  padded=True, 
                  axis=- 1)
STFTMatrix = np.fft.fftshift((np.abs(Zxx)), axes=0)
#print("X\n", STFTMatrix)
FreqAxis=np.linspace(ppmfirst,ppmlast,len(STFTMatrix.T[0])) 

TimeAxis=np.linspace(0,at,len(STFTMatrix[1]) )

fig = plt.figure()
ax=fig.add_subplot(111,projection="3d")
for i in range(STFTMatrix.shape[0]):
    l= [FreqAxis[i]]*STFTMatrix[i].shape[0]
    #print("l", l)
    m= TimeAxis#range(STFTMatrix[i].shape[0])
    #print("m", m)
    n=STFTMatrix[i]
    #print("n", n)
    plt.plot(l,m,n,marker=".",alpha=0.5)
    plt.tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False, labelsize=10)
#ax.axis("off")
#ax.set_xticks(FreqAxis)
#ax.set_yticks(np.linspace(0, at, STFTMatrix.shape[1]))
ax.set_title("12.STFT(3D)")
ax.set_xlabel('$ChemicalShift[ppm]$')
ax.invert_xaxis()
ax.view_init(elev=30, azim=60)
ax.set_ylabel('$Time[sec]$')
filename = dirname + "3Dmatrix" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
fig.savefig(filename)

plt.close('all')

fig = plt.figure()
plt.title("13.STFT(Freqency domain)")
plt.plot(FreqAxis,STFTMatrix)
os.makedirs(dirname, exist_ok=True)
filename = dirname + "STFT(FreqencyDomain)" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
plt.xlabel("ChemicalShift[ppm]")
plt.gca().invert_xaxis()
fig.savefig(filename)

plt.close('all')

dataFreq = STFTMatrix.T
dataSTFTZscore = scipy.stats.zscore(dataFreq[0])
fig = plt.figure()

peaks, properties = find_peaks(dataSTFTZscore,height=0.1,distance=1)
plt.plot(FreqAxis,dataSTFTZscore)
#print("x\n", x)
plt.plot(FreqAxis[peaks], dataSTFTZscore[peaks], "x")
#print("PeakPoint\n", dataSTFTZscore[peaks])
#print("peaks\n", peaks)
plt.title("14.Zscore + Peak detection")
os.makedirs(dirname, exist_ok=True)
filename = dirname + "Zscore_PeaksDetection" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
plt.xlabel("ChemicalShift[ppm]")
plt.gca().invert_xaxis()
fig.savefig(filename)

plt.close('all')

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
        #print("amp\n", amp)
        ctr = params[int(param_range[1])]
        wid = params[int(param_range[2])]
        #print("wid\n",wid)
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
    amp_list = []
    ctr_list = []
    wid_list = []
    baseline_list = []
    for i in range(num_func):
        y = np.zeros_like(Frequency)
        param_range = list(range(3*i,3*(i+1),1))
        amp = params[int(param_range[0])]
        ctr = params[int(param_range[1])]
        wid = params[int(param_range[2])]
        y = y + amp * np.exp( -((Frequency - ctr)/wid)**2) + params[-1]
        y_list.append(y)
        amp_list.append(amp)
        ctr_list.append(ctr)
        wid_list.append(wid)
        baseline_list.append(params[-1])
    with open(datafilename,"a",newline='')as f:
        writer = csv.writer(f)
        writer.writerow(["M0_list"])
        writer.writerow(amp_list)
        writer.writerow(["Center_list"])
        writer.writerow(ctr_list)
        writer.writerow(["Width_list"])
        writer.writerow(wid_list)
        writer.writerow(["Baseline"])
        writer.writerow(baseline_list)
    return y_list

fig = plt.figure()   
guess = []
Frequency = dataFreq[0]
for j in range(len(peaks)):
    guess.append([Frequency[peaks][j], peaks[j], 0.1])
    print("height",Frequency[peaks][j])
    print("peak",peaks[j])    
background = 0
guess_total = []
for i in guess:
    guess_total.extend(i)
guess_total.append(background)

Frequency = dataFreq[0]
x = np.arange(len(Frequency))
param_bounds=([-np.inf,-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf,np.inf])
popt, pcov = curve_fit(funcFreq, x, Frequency, p0=guess_total, bounds=param_bounds)
#print('x\n',x)
#print('Frequency\n',Frequency)
#print('LenFrequency\n',len(Frequency))
#with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
with open(datafilename,"a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow(["FreqAxis_for_fit"])
    writer.writerow(x)
    writer.writerow(["Spectrum_for_fit"])
    writer.writerow(Frequency)

fit = funcFreq(x, *popt)
#print('Fit',len(fit))
#FreqAxis = np.arange(ppmmin, ppmmax, dic['acqus']['SW']/len(Frequency))
#FreqAxis=np.linspace(0,12,len(fit))
FreqAxis=np.linspace(ppmfirst,ppmlast,len(fit)) 
plt.plot(FreqAxis,x)
plt.plot(FreqAxis[peaks], Frequency[peaks], "x")
plt.scatter(FreqAxis, Frequency, s=20)
plt.plot(FreqAxis, fit , ls='-', c='black', lw=1)
#with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
with open(datafilename,"a",newline='')as f:
    writer = csv.writer(f)
    writer.writerow(["FreqAxis_fit"])
    writer.writerow(FreqAxis)
    writer.writerow(["Spectrum_fit"])
    writer.writerow(fit)

y_list = fit_plotFreq(FreqAxis, *popt)
baseline = np.zeros_like(x) + popt[-1]
for n,i in enumerate(y_list):
    plt.fill_between(FreqAxis, i, baseline, facecolor=cm.rainbow(n/len(y_list)), alpha=0.6)
plt.title("15.Fitting(Frequensy domain)")

now = datetime.datetime.now()
#dirname = "FreqFitting_Low/"
os.makedirs(dirname, exist_ok=True)
filename = dirname + "Fitting(FrequensyDomain)" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
plt.xlabel("ChemicalShift[ppm]")
plt.gca().invert_xaxis()
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
        print("M0\n",a)
        W = params[int(param_range[1])]
        print("Weibull coefficient\n",W)
        T2 = params[int(param_range[2])]
        print("T2\n",T2*at/len(x))
        y = y + a*np.exp(-1/W *(x/T2)**W)
        y_list.append(y)
        #print(np.max(y_list[i]))
    with open(datafilename,"a",newline='')as f:
            writer = csv.writer(f)
            writer.writerow(["M0"])
            writer.writerow([a])
            writer.writerow(["WeibullCoefficient"])
            writer.writerow([W])
            writer.writerow(["T2[sec]"])
            writer.writerow([T2*at/len(x)])

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
    #fig=plt.figure()
    #plt.plot(slice)
    print("PeakNo.\n",j+1)
    print("ChemicalshiftNO.\n",peaks[j])
    print("Chemicalshift[ppm]\n",FreqAxis[peaks[j]])
    y = slice
    #print(slice)
    x1 = np.count_nonzero(y)
    x = np.arange(x1)
    with open(datafilename,"a",newline='')as f:
        writer = csv.writer(f)
        writer.writerow([filename])
        writer.writerow(["PeakNo."]) 
        writer.writerow([j+1]) 
        writer.writerow(["ChemicalshiftNO."]) 
        writer.writerow([peaks[j]]) 
        writer.writerow(["Chemicalshift[ppm]"]) 
        writer.writerow([FreqAxis[peaks[j]]]) 
    
    #print(x)
    #print(y)
    #print(guess_total)
    popt, pcov = curve_fit(funcTime, x, y, p0=guess_total, bounds=param_bounds)

    #print(popt)
    #fig=plt.figure()
    fit = funcTime(x, *popt)
    #plt.scatter(x*at/len(x), y, s=20, c='black')
    #plt.plot(x*at/len(x), fit , ls='-', c='black', lw=1)
    
    y_list = fit_plotTime(x, *popt)
    baseline = np.zeros_like(x) + popt[-1]
   # chemicalShift=peaks*((ppmmax-ppmmin)/TD) +ppmmin 
    for n,i in enumerate(y_list):
        fig=plt.figure()
        plt.scatter(x*at/len(x), y, s=20, c='black')
        plt.plot(x*at/len(x), fit , ls='-', c='black', lw=1)
        plt.fill_between(x*at/len(x), i , baseline, facecolor=cm.rainbow(n/len(y_list)), alpha=0.6)
        figtitle = "16.PeakNo"+ str(j+1) +"_Fitting(Time domain)"+ "_" + str(round(FreqAxis[peaks[j]],2)) +"[ppm]" +"_T2[sec]"+ str(round(popt[2]*at/len(x),4))
        plt.title(figtitle)
        plt.xlabel("Time[sec]") 
    plt.close('all')

    now = datetime.datetime.now()
    #dirname = "FreqFitting_Low/"
    os.makedirs(dirname, exist_ok=True)
    filename = dirname + "PeakNo"+ str(j+1) + "_Fitting(TimeDomain)_" +now.strftime('%Y%m%d_%H%M%S') + "img.png"
    fig.savefig(filename)

    timefit_y = ["TimeFit_y"]
    timefit_x = ["TimeFit_x"]
    #with open("TransferSimulationLowSpectrum.csv","a",newline='')as f:
    with open(datafilename,"a",newline='')as f:
        writer = csv.writer(f)
        writer.writerow(timefit_y)
        writer.writerow(y_list[0])
        writer.writerow(["Normalized_timefit_y"])
        writer.writerow(y_list[0]/max(y_list[0]))
        writer.writerow(timefit_x)
        writer.writerow(x*at/len(x))


