import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy as sci
import copy
from Functions.f_SignalProcFuncLibs import *
from scipy.signal import welch
from scipy.integrate import simps


str_DataPath = 'Data/MatData/'  # path file where the data is stored
str_FileName = 'tachogram.mat'  # Name of the File
str_MyRed = '#7B241C'
str_MyBlue = '#0C649E'
v_allData = sio.loadmat(str_DataPath + str_FileName)  # Load .mat data
v_Taco = np.double(v_allData['v_Taco'][0])
v_Time_Taco = np.double(v_allData['v_Time_Taco'][0])

function = sci.interpolate.CubicSpline(v_Time_Taco, v_Taco, bc_type='natural')

d_NewSampleRate = 4
v_Time_TacoInter = np.arange(0, v_Time_Taco[-1], 1 / d_NewSampleRate)
v_TacoInter = function(v_Time_TacoInter)

fig, ax = plt.subplots()
ax.plot(v_Time_TacoInter, v_TacoInter, linewidth=1, color=str_MyRed, label='Interpolado')
ax.plot(v_Time_Taco, v_Taco, linewidth=1, color=str_MyBlue, label='Original')
ax.grid(linewidth=0.65, linestyle=':', which='both', color='k')
ax.legend()

d_WindSec = 20
d_stepSec = 5
d_WindSam = d_WindSec * d_NewSampleRate
d_stepSam = d_stepSec * d_NewSampleRate
n_win = int(len(v_Time_TacoInter) / d_stepSec)

d_indexStart = 0
d_indexEnd = d_indexStart + d_WindSam

v_LF = []
v_HF = []
while d_indexEnd <= len(v_TacoInter):
    i_dataWind = v_TacoInter[d_indexStart:d_indexEnd]
    freqs, psd = welch(i_dataWind, d_NewSampleRate, nperseg=80,nfft=len(i_dataWind))
    freq_res = freqs[1] - freqs[0]
    # Find the closest indices of band in frequency vector


    idx_band_LF = np.logical_and(freqs >= 0.04, freqs <= 0.15)
    idx_band_HF = np.logical_and(freqs >= 0.15, freqs <= 0.4)
    # Integral approximation of the spectrum using Simpson's rule.
    d_LF = simps(psd[idx_band_LF], dx=freq_res)
    d_HF = simps(psd[idx_band_HF], dx=freq_res)

    v_LF.append(d_LF)
    v_HF.append(d_HF)

    d_indexStart = d_indexStart + d_stepSam
    d_indexEnd = d_indexStart + d_WindSam

##
fig, ax = plt.subplots()
ax.plot(v_LF, linewidth=1, color=str_MyRed, label='LF')
ax.plot(v_HF, linewidth=1, color=str_MyBlue, label='HF')
ax.grid(linewidth=0.65, linestyle=':', which='both', color='k')
ax.legend()