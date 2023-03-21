import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.signal as sig

str_DataPath = 'Data/MatData/'  # path file where the data is stored
str_FileName = 'ECGData.mat'  # Name of the File
v_allData = sio.loadmat(str_DataPath + str_FileName)  # Load .mat data
v_ECGSig = np.double(v_allData['m_Data'][0])
d_SampleRate = v_allData['s_Freq']
str_MyRed = '#7B241C'
str_MyBlue = '#0C649E'
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

fig, ax = plt.subplots(4,1,sharex=True)
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

fig, ax = plt.subplots(2,1, sharex=True)
fig.suptitle('Tachogram')
ax[0].plot(v_TimeArray, v_ECGSig, linewidth=1, color=str_MyBlue, label='ECG R points')
ax[0].plot(v_TimeArray[v_PeaksInd], v_ECGSig[v_PeaksInd], '.', color=str_MyRed)
ax[0].grid(linewidth=0.65, linestyle=':', which='both', color='k')

ax[1].plot(v_Time_Taco, v_Taco, linewidth=1, color=str_MyBlue, label='Tachogram')
ax[1].grid(linewidth=0.65, linestyle=':', which='both', color='k')
ax[1].set_ylabel('R-R Time (Seconds)')
ax[1].set_xlabel('Time (Seconds)')

sio.savemat(str_DataPath + 'tachogram.mat', mdict={'v_Taco': v_Taco,
                                          'v_Time_Taco': v_Time_Taco})
