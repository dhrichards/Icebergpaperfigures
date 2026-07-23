#%%
import numpy as np
from matplotlib import pyplot as plt


berg_heights = np.array([350,400,450,500,550])

berg_dts = 1.0

berg_fail_itsdt1 = np.array([34,13,9,10,5])

berg_fail_its0p1 = np.array([131,55,33,22,18])
bergs_fail_times = (berg_fail_its0p1)*0.1

berg_heightstol5 = np.array([325,350,400,425,450,475,500,525,550,575])
berg_fail_itstol5 = np.array([121,129,54,48,31,29,19,15,17,16])


berg_heights_l5 = np.array([350,400,450,500,550])
berg_fail_its_l5 = np.array([34,19,13,11,14])

berg_heights_L2000 = np.array([250,300,350,400,450,500,550,600,650])
berg_fail_its_L2000 = np.array([475,362,294,284,265,270,264,279,282])

berg_heights_L8000 = np.array([350,450,500,550])
berg_fail_its_L8000 = np.array([194,58,15,5])

berg_heights_L16000 = np.array([350,450,500,550])#400
berg_fail_its_L16000 = np.array([418,33,25,12])#21

berg_heights_L8000_l5 = np.array([350,400,450,500])
berg_fail_its_L8000_l5 = np.array([181,86,24,14])

berg_heights_aspect_ratio_5 = np.array([350,400,450,500,550])
berg_fail_its_aspect_ratio_5 = np.array([498,285,158,103,71])

berg_heights_aspect_ratio_10 = np.array([350,400,450,500])
berg_fail_its_aspect_ratio_10 = np.array([175, 55, 23,11])

berg_heights_ar_10_no_T = np.array([300,350,400,450,500,550])
berg_fail_its_ar_10_no_T = np.array([113,39,15,6,3,48])

berg_heights_L8000_no_T = np.array([350,400,450,500,550])
berg_fail_its_L8000_no_T = np.array([98, 68, 18, 42, 38])

berg_heights_ar20_no_T = np.array([350,400,450,500,550])
berg_fail_its_ar20_no_T = np.array([2, 86, 30,46,27])

berg_heights_L4000_no_T = np.array([350,400,450,500,550])
berg_fail_its_L4000_no_T = np.array([12,15,16,19,21])




## plot iceberg time to failure vs height
fig, ax = plt.subplots(1,1,figsize=(6,5))
# ax.scatter(berg_heights, bergs_fail_times, marker='o', label='Icebergs')

# ax.scatter(berg_heightstol5,berg_fail_itstol5*0.1, marker='o', label='Icebergs')


# ax.scatter(berg_heights_L2000,berg_fail_its_L2000*0.1, marker='o', label='L=2000')
# ax.scatter(berg_heights_L8000,berg_fail_its_L8000*0.1, marker='o', label='L=8000')
# ax.scatter(berg_heights_L16000,berg_fail_its_L16000*0.1, marker='o', label='L=16000')
# ax.scatter(berg_heights_L8000_l5,berg_fail_its_L8000_l5*0.1, marker='o', label='L=8000, l=5')
# ax.scatter(berg_heights_aspect_ratio_5,berg_fail_its_aspect_ratio_5*0.1, marker='o', label='Aspect ratio 5')
# ax.scatter(berg_heights_aspect_ratio_10,berg_fail_its_aspect_ratio_10*0.1, marker='o', label='Aspect ratio 10')
ax.scatter(berg_heights_ar_10_no_T,berg_fail_its_ar_10_no_T*0.1, marker='o', label='Aspect ratio 10, no T')
ax.scatter(berg_heights_L8000_no_T,berg_fail_its_L8000_no_T*0.1, marker='o', label='L=8000, no T')
ax.scatter(berg_heights_ar20_no_T,berg_fail_its_ar20_no_T*0.1, marker='o', label='Aspect ratio 20, no T')
ax.scatter(berg_heights_L4000_no_T,berg_fail_its_L4000_no_T*0.1, marker='o', label='L=4000, no T')

ax.set_yscale('log')
ax.set_xscale('log')

# polyfit for height>340
# coeffs = np.polyfit(np.log(berg_heightstol5), np.log(0.1*berg_fail_itstol5), 1)
# print('Polyfit coefficients for iceberg time to failure vs height:', coeffs)
x_fit = np.linspace(300, 700, 100)
# y_fit = np.exp(coeffs[1]) * x_fit**coeffs[0]
# ax.plot(x_fit, y_fit, label=f'Fit: time ~ height^{coeffs[0]:.2f}', ls='--') 
ax.set_xlabel('Height (m)')
ax.set_ylabel('Time to failure (days)')

# coeffs_L2000 = np.polyfit(np.log(berg_heights_L2000), np.log(0.1*berg_fail_its_L2000), 1)
# print('Polyfit coefficients for L=2000 time to failure vs height:', coeffs_L2000)
# y_fit_L2000 = np.exp(coeffs_L2000[1]) * x_fit**coeffs_L2000[0]
# ax.plot(x_fit, y_fit_L2000, label=f'L=2000 Fit: time ~ height^{coeffs_L2000[0]:.2f}', ls='--')

# coeffs_L8000 = np.polyfit(np.log(berg_heights_L8000), np.log(0.1*berg_fail_its_L8000), 1)
# print('Polyfit coefficients for L=8000 time to failure vs height:', coeffs_L8000,    8000)
# y_fit_L8000 = np.exp(coeffs_L8000[1]) * x_fit**coeffs_L8000[0]
# ax.plot(x_fit, y_fit_L8000, label=f'L=8000 Fit: time ~ height^{coeffs_L8000[0]:.2f}', ls='--')      

# coeffs_L16000 = np.polyfit(np.log(berg_heights_L16000), np.log(0.1*berg_fail_its_L16000), 1)
# print('Polyfit coefficients for L=16000 time to failure vs height:', coeffs_L16000,    16000)
# y_fit_L16000 = np.exp(coeffs_L16000[1]) * x_fit**coeffs_L16000[0]
# ax.plot(x_fit, y_fit_L16000, label=f'L=16000 Fit: time ~ height^{coeffs_L16000[0]:.2f}', ls='--')


# coeffs_L8000_l5 = np.polyfit(np.log(berg_heights_L8000_l5), np.log(0.1*berg_fail_its_L8000_l5), 1)
# print('Polyfit coefficients for L=8000, l=5 time to failure vs height:', coeffs_L8000_l5,    8000)
# y_fit_L8000_l5 = np.exp(coeffs_L8000_l5[1]) * x_fit**coeffs_L8000_l5[0]
# ax.plot(x_fit, y_fit_L8000_l5, label=f'L=8000, l=5 Fit: time ~ height^{coeffs_L8000_l5[0]:.2f}', ls='--')   

# coeffs_aspect_ratio_5 = np.polyfit(np.log(berg_heights_aspect_ratio_5), np.log(0.1*berg_fail_its_aspect_ratio_5), 1)
# print('Polyfit coefficients for aspect ratio 5 time to failure vs height:', coeffs_aspect_ratio_5,    8000)
# y_fit_aspect_ratio_5 = np.exp(coeffs_aspect_ratio_5[1]) * x_fit**coeffs_aspect_ratio_5[0]
# ax.plot(x_fit, y_fit_aspect_ratio_5, label=f'Aspect ratio 5 Fit: time ~ height^{coeffs_aspect_ratio_5[0]:.2f}', ls='--')


# coeffs_aspect_ratio_10 = np.polyfit(np.log(berg_heights_aspect_ratio_10), np.log(0.1*berg_fail_its_aspect_ratio_10), 1)
# print('Polyfit coefficients for aspect ratio 10 time to failure vs height:', coeffs_aspect_ratio_10,    8000)
# y_fit_aspect_ratio_10 = np.exp(coeffs_aspect_ratio_10[1]) * x_fit**coeffs_aspect_ratio_10[0]
# ax.plot(x_fit, y_fit_aspect_ratio_10, label=f'Aspect ratio 10 Fit: time ~ height^{coeffs_aspect_ratio_10[0]:.2f}', ls='--') 

coeffs_ar_5_no_T = np.polyfit(np.log(berg_heights_ar_10_no_T[:-1]), np.log(0.1*berg_fail_its_ar_10_no_T[:-1]), 1)
print('Polyfit coefficients for aspect ratio 5, no T time to failure vs height:', coeffs_ar_5_no_T,    8000)
y_fit_ar_5_no_T = np.exp(coeffs_ar_5_no_T[1]) * x_fit**coeffs_ar_5_no_T[0]
ax.plot(x_fit, y_fit_ar_5_no_T, label=f'Aspect ratio 10, no T Fit: time ~ height^{coeffs_ar_5_no_T[0]:.2f}', ls='--')    
#legend outside
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title('Time to failure vs height for icebergs')



