import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.signal as sig
from Functions.f_SignalProcFuncLibs import *

str_DataPath = 'Data/MatData/'  # path file where the data is stored
str_FileName = 'ECGData.mat'  # Name of the File
v_allData = sio.loadmat(str_DataPath + str_FileName)  # Load .mat data
v_ECGSig = np.double(v_allData['m_Data'][0])
d_SampleRate = v_allData['s_Freq']
str_MyRed = '#7B241C'
str_MyBlue = '#0C649E'



m_ConvMat, v_TimeArray, v_FreqTestHz = f_GaborTFTransform(v_ECGSig, d_SampleRate, 1, 50, 0.25, 9)
v_TimeArray = v_TimeArray[0]

##
fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)

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
# Cómo re muestrearla?
# Cómo filtrarla?
# Qué es la frecuencia de muestreo?
# Cómo funciona el algoritmo de Pan Tompkins?