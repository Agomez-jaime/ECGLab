import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.signal as sig

str_DataPath = 'Data/MatData/'  # path file where the data is stored
str_FileName = 'tachogram.mat'  # Name of the File
str_MyRed = '#7B241C'
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

fig, ax = plt.subplots(5,1,sharex=True)

fig.suptitle('HRV stats')
ax[0].plot(v_TimeArray, v_MEANNN, linewidth=2, color=str_MyRed, label='RawData')
ax[1].plot(v_TimeArray, v_STDNN, linewidth=2, color=str_MyRed, label='Derivative')
ax[2].plot(v_TimeArray, v_NN50, linewidth=2, color=str_MyRed, label='Square')
ax[3].plot(v_TimeArray, v_PNN50, linewidth=2, color=str_MyRed, label='Square')
ax[4].plot(v_Time_Taco, v_Taco, linewidth=2, color=str_MyRed, label='Square')

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