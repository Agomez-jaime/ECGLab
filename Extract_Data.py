##

import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from Functions.f_TRC_Reader import *
from Functions.f_SignalProcFuncLibs import *

str_DataPath = 'Data/TRC_File/' #path file where the data is stored
str_OutPath = 'Data/MatData/'   #path file where the data will be stored

if not os.path.isdir(str_OutPath):  # Create the path if this doesn't exist
    os.mkdir(str_OutPath)

str_ReadName = str_DataPath + 'ECG.TRC' #Name of the file
str_SaveName = str_OutPath + 'ECGData' #Nome of teh new file

str_ChannStr = ['ECG1+'] #Label of the channel

st_TRCHead = f_GetTRCHeader(str_ReadName)  # Function that extracts the TCR header
d_SampleRate = st_TRCHead['RecFreq']  # Sample frequency
print(f'Sampling rate: {d_SampleRate}')
m_AllData = f_GetSignalsTRC(str_ReadName, str_ChannStr)  # Function that extracts the TCR header data


m_Data = m_AllData[0]


st_Filt = f_GetIIRFilter(d_SampleRate, [1, 59.5], [0.95, 60]) #Infinite response filter (band reject)
v_DataFilt = f_IIRBiFilter(st_Filt, m_Data)


v_TimeArray = np.arange(0, np.size(v_DataFilt)) / d_SampleRate  # Time values

str_MyRed = '#7B241C'
str_MyBlue = '#0C649E'

fig, ax = plt.subplots()
ax.plot(v_TimeArray, m_AllData[0], linewidth=0.85, color=str_MyRed, label='RawData')
ax.plot(v_TimeArray, v_DataFilt, linewidth=0.85, color=str_MyBlue, label='Filtered Data')
ax.grid(linewidth=0.65, linestyle=':', which='both', color='k')
ax.set_xlabel('Time (Seconds)')
plt.legend()

sio.savemat(str_SaveName + '.mat', mdict={'m_Data': v_DataFilt,
                                          'v_ChanNames': str_ChannStr,
                                          's_Freq': d_SampleRate})

print(f'---------------------------------------------')
