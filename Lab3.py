# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:29:19 2023

@author: Andrea Gómez y Sebastián Manrique

Créditos: Jose Silva
"""
##
import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.signal as sig
import struct as st
import os
from scipy.interpolate import interp1d
import scipy.integrate as int
from Functions.f_TRC_Reader import *
from Functions.f_Graphing import *
from Functions.f_SignalProcFuncLibs import *

str_DataPath = 'Data/ECG/' #path file where the data is stored
str_OutPath = 'Data/OutData/'   #path file where the data will be stored

if not os.path.isdir(str_OutPath):  # Create the path if this doesn't exist
    os.mkdir(str_OutPath)

#Punto 1
file_names = ["001N1_ECG", "002N7_ECG", "004N8_ECG",
              "005N1_ECG", "006N1_ECG"]
"""
for name in file_names:
    str_ReadName = str_DataPath + name + '.mat' #Name of the file
    str_SaveName = str_OutPath + name +'Data' #Nome of teh new file
    
    '''
    ecg_data = sio.loadmat(str_ReadName)
    d_SampleRate = ecg_data["s_FsHz"][0][0]
    m_Data = ecg_data["v_ECGSig"][0]
    
    st_Filt = f_GetIIRFilter(d_SampleRate, [1, 59.5], [0.95, 60]) #Infinite response filter (band reject)
    v_DataFilt = f_IIRBiFilter(st_Filt, m_Data)


    v_TimeArray = np.arange(0, np.size(v_DataFilt)) / d_SampleRate  # Time values
    '''
    v_TimeArray, v_DataFilt, m_Data, d_SampleRate = f_preprocessing(str_ReadName, [1, 59.5], [0.95, 60])
    
    str_MyRed = '#7B241C'
    str_MyBlue = '#0C649E'

    fig, ax = plt.subplots()
    fig.suptitle(name)
    ax.plot(v_TimeArray, m_Data, linewidth=0.85, color=str_MyRed, label='RawData')
    ax.plot(v_TimeArray, v_DataFilt, linewidth=0.85, color=str_MyBlue, label='Filtered Data')
    ax.grid(linewidth=0.65, linestyle=':', which='both', color='k')
    ax.set_xlabel('Time (Seconds)')
    plt.legend()
    plt.show()
    sio.savemat(str_SaveName + '.mat', mdict={'m_Data': v_DataFilt,
                                              's_Freq': d_SampleRate})

    print(f'---------------------------------------------')

#Punto 2
for name in file_names:
    str_DataPath = 'Data/OutData/'  # path file where the data is stored
    str_FileName = name + 'Data.mat'  # Name of the File
    
    str_MyRed = '#7B241C'
    str_MyBlue = '#0C649E'
    
    '''
    v_allData = sio.loadmat(str_DataPath + str_FileName)  # Load .mat data
    d_SampleRate = v_allData['s_Freq']
    end = d_SampleRate[0][0]*15
    v_ECGSig = v_allData['m_Data'][0][0:end] 
    v_ECGSig = np.double(v_ECGSig)
    
    
    
    m_ConvMat, v_TimeArray, v_FreqTestHz = f_GaborTFTransform(v_ECGSig, d_SampleRate, 1, 50, 0.25, 9)
    v_TimeArray = v_TimeArray[0]
    '''
    m_ConvMat, v_TimeArray, v_FreqTestHz, v_ECGSig = f_convolution(str_DataPath, str_FileName)
    ##
    fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    fig.suptitle(name)
    #v_ECGSig_w = 0 : 15*s_FsHz
    # Graficamos la señal x1
    ax[0].plot(v_TimeArray, v_ECGSig, linewidth=0.75)
    ax[0].set_ylabel("ECG", fontsize=15)
    ax[0].grid(1)
    
    # Graficamos la matriz resultante en escala de colores
    # ConvMatPlot = ConvMat
    m_ConvMatPlot = np.abs(m_ConvMat)
    immat = ax[1].imshow(m_ConvMatPlot, cmap='hot', interpolation='none',
                               origin='lower', aspect='auto',
                               extent=[v_TimeArray[0], v_TimeArray[-1],
                                       v_FreqTestHz[0], v_FreqTestHz[-1]],
                               vmin=-0, vmax=15000)
    
    immat.set_clim(np.min(m_ConvMatPlot) , np.max(m_ConvMatPlot) * 0.05 )
    ax[1].set_xlabel("Time (sec)", fontsize=15)
    ax[1].set_ylabel("Freq (Hz)", fontsize=15)
    ax[1].set_xlim([v_TimeArray[0], v_TimeArray[-1]])
    fig.colorbar(immat, ax=ax[1])

#Punto 3
for name in file_names:
    str_DataPath = 'Data/OutData/'  # path file where the data is stored
    str_FileName = name + 'Data.mat'  # Name of the File
    
    str_MyRed = '#7B241C'
    str_MyBlue = '#0C649E'

    v_allData = sio.loadmat(str_DataPath + str_FileName)  # Load .mat data
    v_ECGSig = np.double(v_allData['m_Data'][0])
    d_SampleRate = v_allData['s_Freq']

    '''
    v_TimeArray = np.arange(np.size(v_ECGSig)) / d_SampleRate  # Time values
    v_TimeArray = v_TimeArray[0]
    
    v_ECGFiltDiff = np.zeros(np.size(v_ECGSig))
    v_ECGFiltDiff[1:] = np.diff(v_ECGSig)  # Se extrae la derivada
    v_ECGFiltDiff[0] = v_ECGFiltDiff[1]
    v_ECGFiltDiffSqrt = v_ECGFiltDiff ** 2  # Atenuar lo pequeña y ampliar lo que es grande, acumula los dos picos anteriores en uno solo
    
    s_AccSumWinSizeSec = 0.03  # Ventana de 30 ms
    s_AccSumWinHalfSizeSec = s_AccSumWinSizeSec / 2.0  # Toma la mitad del intervalo
    s_AccSumWinHalfSizeSam = int(np.round(s_AccSumWinHalfSizeSec * d_SampleRate))  # Nos da el numero de puntos de la ventana
    v_ECGFiltDiffSqrtSum = np.zeros(np.size(v_ECGFiltDiffSqrt))  # Se inicializa el arreglo donde guardaremos la suma de la ventana
    
    
    for s_Count in range(np.size(v_ECGFiltDiffSqrtSum)):
        s_FirstInd = s_Count - s_AccSumWinHalfSizeSam
        s_LastInd = s_Count + s_AccSumWinHalfSizeSam
        if s_FirstInd < 0:
            s_FirstInd = 0
        if s_LastInd >= np.size(v_ECGFiltDiffSqrtSum):
            s_LastInd = np.size(v_ECGFiltDiffSqrtSum)
        v_ECGFiltDiffSqrtSum[s_Count] = np.mean(v_ECGFiltDiffSqrt[s_FirstInd:s_LastInd + 1])
    '''
    
    v_TimeArray, v_ECGSig, v_ECGFiltDiff, v_ECGFiltDiffSqrt, v_ECGFiltDiffSqrtSum = f_RR(str_DataPath, str_FileName)
    
    fig, ax = plt.subplots(4,1,sharex=True)
    fig.suptitle(name)
    ax[0].plot(v_TimeArray, v_ECGSig, linewidth=1, color=str_MyRed, label='RawData')
    ax[1].plot(v_TimeArray, v_ECGFiltDiff, linewidth=1, color=str_MyRed, label='Derivative')
    ax[2].plot(v_TimeArray, v_ECGFiltDiffSqrt, linewidth=1, color=str_MyRed, label='Square')
    ax[3].plot(v_TimeArray, v_ECGFiltDiffSqrtSum, linewidth=1, color=str_MyRed, label='Square')
    ax[0].grid(linewidth=0.65, linestyle=':', which='both', color='k')
    ax[1].grid(linewidth=0.65, linestyle=':', which='both', color='k')
    ax[2].grid(linewidth=0.65, linestyle=':', which='both', color='k')
    ax[3].grid(linewidth=0.65, linestyle=':', which='both', color='k')
    ax[3].set_xlabel('Time (Seconds)')
    
    ax[0].set_yticklabels([])
    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    
    plt.subplots_adjust(wspace=0, hspace=0)
    '''
    ##
    v_PeaksInd = sig.find_peaks(v_ECGFiltDiffSqrtSum)
    v_Peaks = v_ECGFiltDiffSqrtSum[v_PeaksInd[0]]
    s_PeaksMean = np.mean(v_Peaks)
    s_PeaksStd = np.std(v_Peaks)
    
    s_MinTresh = s_PeaksMean + 1 * s_PeaksStd
    s_MaxTresh = s_PeaksMean + 8 * s_PeaksStd
    s_QRSInterDurSec = 0.2
    
    s_MinDurSam = np.round(s_QRSInterDurSec * d_SampleRate)
    v_PeaksInd, _ = sig.find_peaks(v_ECGFiltDiffSqrtSum, height=[s_MinTresh, s_MaxTresh], distance=s_MinDurSam)
    
    # Corregir esa identificación de picos corridos
    s_QRSPeakAdjustHalfWinSec = 0.05
    s_QRSPeakAdjustHalfWinSam = int(np.round(s_QRSPeakAdjustHalfWinSec * d_SampleRate))
    for s_Count in range(np.size(v_PeaksInd)):
        s_Ind = v_PeaksInd[s_Count]
        s_FirstInd = s_Ind - s_QRSPeakAdjustHalfWinSam
        s_LastInd = s_Ind + s_QRSPeakAdjustHalfWinSam
        if s_FirstInd < 0:
            s_FirstInd = 0
        if s_LastInd >= np.size(v_ECGSig):
            s_LastInd = np.size(v_ECGSig)
        v_Aux = v_ECGSig[s_FirstInd:s_LastInd + 1]
        v_Ind1 = sig.find_peaks(v_Aux)
        if np.size(v_Ind1[0]) == 0:
            continue
        s_Ind2 = np.argmax(v_Aux[v_Ind1[0]])
        s_Ind = int(v_Ind1[0][s_Ind2])
        v_PeaksInd[s_Count] = s_FirstInd + s_Ind
    
    v_Taco = np.diff(v_PeaksInd) / d_SampleRate
    v_Taco = v_Taco[0]
    v_Time_Taco = v_TimeArray[v_PeaksInd[1:]]
    '''
    v_TimeArray, v_ECGSig, v_PeaksInd, v_Time_Taco, v_Taco = f_taco(v_ECGFiltDiffSqrtSum, d_SampleRate, v_ECGSig, v_TimeArray)
    
    fig, ax = plt.subplots(2,1, sharex=True)
    fig.suptitle('Tachogram'+name)
    ax[0].plot(v_TimeArray, v_ECGSig, linewidth=1, color=str_MyBlue, label='ECG R points')
    ax[0].plot(v_TimeArray[v_PeaksInd], v_ECGSig[v_PeaksInd], '.', color=str_MyRed)
    ax[0].grid(linewidth=0.65, linestyle=':', which='both', color='k')
    
    ax[1].plot(v_Time_Taco, v_Taco, linewidth=1, color=str_MyBlue, label='Tachogram')
    ax[1].grid(linewidth=0.65, linestyle=':', which='both', color='k')
    ax[1].set_ylabel('R-R Time (Seconds)')
    ax[1].set_xlabel('Time (Seconds)')
    
    sio.savemat(str_DataPath + 'tachogram' + str(name)+ '.mat', mdict={'v_Taco': v_Taco,
                                              'v_Time_Taco': v_Time_Taco})

#Punto 4 y 5
for name in file_names:
    str_DataPath = 'Data/OutData/'  # path file where the data is stored
    str_FileName = 'tachogram' + str(name)+ '.mat' # Name of the File
    
    str_MyRed = '#7B241C'
    
    '''
    v_allData = sio.loadmat(str_DataPath + str_FileName)  # Load .mat data
    v_Taco = np.double(v_allData['v_Taco'][0])
    v_Time_Taco = np.double(v_allData['v_Time_Taco'][0])
    
    v_MEANNN, v_STDNN, v_NN50, v_PNN50 = [], [], [], []
    
    d_WindSec = 20
    d_stepSec = 5
    
    i_TimeStart = 0
    i_TimeEnd = i_TimeStart + d_WindSec
    
    while i_TimeEnd <= v_Time_Taco[-1]:
        v_TacoIndx = (v_Time_Taco < i_TimeEnd) & (v_Time_Taco > i_TimeStart)
        v_TacoWind = v_Taco[v_TacoIndx]
    
        v_MEANNN.append(np.mean(v_TacoWind))
        v_STDNN.append((np.std(v_TacoWind)))
    
        # NN50
        i_NN50 = len(np.where(np.abs(v_TacoWind[1:] - v_TacoWind[:1]) > 0.05)[0])
        i_PNN50 = i_NN50 / len(v_TacoWind) * 100
        v_NN50.append(i_NN50)
        v_PNN50.append(i_PNN50)
    
        i_TimeStart = i_TimeStart + d_stepSec
        i_TimeEnd = i_TimeStart + d_WindSec
    
    v_TimeArray = np.arange(len(v_PNN50))*d_stepSec
    '''
    v_TimeArray, v_MEANNN, v_STDNN, v_NN50, v_PNN50, v_Taco_stats, v_Time_Taco_stats = f_stats(str_DataPath, str_FileName)
    
    fig, ax = plt.subplots(5,1,sharex=True)
    
    fig.suptitle('HRV stats '+name)
    ax[0].plot(v_TimeArray_stats, v_MEANNN, linewidth=2, color=str_MyRed, label='RawData')
    ax[1].plot(v_TimeArray_stats, v_STDNN, linewidth=2, color=str_MyRed, label='Derivative')
    ax[2].plot(v_TimeArray_stats, v_NN50, linewidth=2, color=str_MyRed, label='Square')
    ax[3].plot(v_TimeArray_stats, v_PNN50, linewidth=2, color=str_MyRed, label='Square')
    ax[4].plot(v_Time_Taco_stats, v_Taco_stats, linewidth=2, color=str_MyRed, label='Square')
    
    ax[0].set_ylabel('Mean R-R (Sec)')
    ax[1].set_ylabel('Std R-R (Sec)')
    ax[2].set_ylabel('NN50 (Count)')
    ax[3].set_ylabel('pNN5 (%)')
    ax[4].set_ylabel('Tacograma')
    
    ax[0].grid(linewidth=0.65, linestyle=':', which='both', color='k')
    ax[1].grid(linewidth=0.65, linestyle=':', which='both', color='k')
    ax[2].grid(linewidth=0.65, linestyle=':', which='both', color='k')
    ax[3].grid(linewidth=0.65, linestyle=':', which='both', color='k')
    ax[4].grid(linewidth=0.65, linestyle=':', which='both', color='k')
    ax[4].set_xlabel('Time (Seconds)')
    plt.show()


#Punto 6

str_DataPath = 'Data/ECG/' #path file where the data is stored
str_OutPath = 'Data/OutData/'   #path file where the data will be stored
for name in file_names:
    str_ReadName = str_DataPath + name + '.mat' #Name of the file
    str_SaveName = str_OutPath + name +'Data' #Nome of teh new file


    ecg_data = sio.loadmat(str_ReadName)
    d_SampleRate = ecg_data["s_FsHz"][0][0]

    #d_SampleRate = ecg_data["s_FsHz"]
    hyp = ecg_data["v_HypCode"] #time:29730, len: 992 time y code
    m_Data = ecg_data["v_ECGSig"][0] #len:3699976
    sts_sleep = HypnogramAverage(m_Data, hyp)
    mean, std, nn, pnn = [], [], [], []
    for i in range(6): #para 8 hay que bajar los coeficientes del filtro para que pase el padlen
        #print(i, [sts_sleep[i]])
        d_SampleRate = int(d_SampleRate)
        sio.savemat(str_DataPath + 'sleep_states' + str(i) + '.mat', mdict={'v_ECGSig': [sts_sleep[i]],
                                                                            's_FsHz': [[d_SampleRate]]})

        #Filtrado
        str_ReadName = str_DataPath + 'sleep_states' + str(i) + '.mat' # Name of the file
        str_SaveName = str_OutPath +  'sleep_states' + str(i) + 'Data'  # Nome of teh new file
        v_TimeArray, v_DataFilt, m_Data, d_SampleRate = f_preprocessing(str_ReadName, [1, 59.5], [0.95, 60])
        sio.savemat(str_SaveName + '.mat', mdict={'m_Data': [v_DataFilt],
                                                  's_Freq': np.array([[d_SampleRate]])})
        d_SampleRate = np.array([[d_SampleRate]])
        #RR y Tacogram
        str_DataPath = 'Data/OutData/'  # path file where the data is stored
        str_FileName = 'sleep_states' + str(i) + 'Data.mat'  # Name of the File
        v_TimeArray, v_ECGSig, v_ECGFiltDiff, v_ECGFiltDiffSqrt, v_ECGFiltDiffSqrtSum = f_RR(str_DataPath, str_FileName)
        v_TimeArray, v_ECGSig, v_PeaksInd, v_Time_Taco_state, v_Taco_state = f_taco(v_ECGFiltDiffSqrtSum, d_SampleRate, v_ECGSig,
                                                                        v_TimeArray)
        sio.savemat(str_DataPath + 'tachogram' + str(i) + str(name) + '.mat', mdict={'v_Taco': list([v_Taco_state]),
                                                           'v_Time_Taco': list([v_Time_Taco_state])})

        #Stats
        str_DataPath = 'Data/OutData/'  # path file where the data is stored
        str_FileName = 'tachogram' + str(i) + str(name) + '.mat'  # Name of the File
        v_TimeArray, v_MEANNN, v_STDNN, v_NN50, v_PNN50, v_Taco_state, v_Time_Taco_state = f_stats(str_DataPath, str_FileName)

    print(v_MEANNN, v_STDNN, v_NN50, v_PNN50) #[1.1744999999999999] [0.3205397791226543] [13] [81.25]
"""
#Punto 7
lf_t = []
hf_t = []
for i in range(6):
    for name in file_names:
        str_DataPath = 'Data/ECG/'  # path file where the data is stored
        str_OutPath = 'Data/OutData/'  # path file where the data will be stored
        str_ReadName = str_DataPath + name + '.mat' #Name of the file
        str_SaveName = str_OutPath + name +'Data' #Nome of teh new file

        ecg_data = sio.loadmat(str_ReadName)
        d_SampleRate = ecg_data["s_FsHz"][0][0]

        str_DataPath = 'Data/OutData/'  # path file where the data is stored
        str_FileName = 'tachogram' + str(i) + str(name) + '.mat'  # Name of the File

        v_allData = sio.loadmat(str_DataPath + str_FileName)  # Load .mat data
        v_Taco = np.double(v_allData['v_Taco'][0])
        v_Time_Taco = np.double(v_allData['v_Time_Taco'][0])

        interp_func = interp1d(v_Time_Taco, v_Taco)

        time = np.arange(v_Time_Taco[0], v_Time_Taco[-1], 0.1)

        new_taco = interp_func(time)
        low_bound = 0
        up_bound = 30*d_SampleRate

        lf = []
        hf = []
        time = []
        while 1:
            if up_bound > len(new_taco):
                up_bound = len(new_taco)
                window_HFO = new_taco[low_bound:up_bound]

                # Análisis espectral de potencia
                l = len(window_HFO)
                v_freq, v_psd = sig.welch(window_HFO, d_SampleRate, nfft=l)
                interp_bands = interp1d(v_freq, v_psd)

                lf_val = int.quad(interp_bands, 0.04, 0.15)
                hf_val = int.quad(interp_bands, 0.15, 0.45)

                lf.append(lf_val)
                hf.append(hf_val)
                time.append(low_bound)

                break

            window_HFO = new_taco[low_bound:up_bound]

            # Análisis espectral de potencia
            l = len(window_HFO)
            v_freq, v_psd = sig.welch(window_HFO, d_SampleRate, nfft=l)
            interp_bands = interp1d(v_freq, v_psd)

            lf_val = int.quad(interp_bands, 0.04, 0.15)
            hf_val = int.quad(interp_bands, 0.15, 0.45)

            lf.append(lf_val)
            hf.append(hf_val)
            time.append(low_bound)

            low_bound += 30*d_SampleRate
            up_bound += 30*d_SampleRate

        plt.figure()
        plt.plot(time, lf)
        plt.plot(time, hf)
        plt.title(name + " " + i)
        plt.show()

        lf_t += lf
        hf_t += hf
    av_lf = np.average(lf)
    av_hf = np.average(hf)

    plt.figure()
    plt.hist([av_lf, av_hf, av_lf/av_hf])
    plt.title('Estado: ' + i)
    plt.show()
