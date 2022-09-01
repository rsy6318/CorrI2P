import numpy as np
import os
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

base_path='./2022-01-07/result_error_2.00'

total_num=len(os.listdir(base_path))//4

t_error_set=[]
r_error_set=[]

for i in range(total_num):
    t_error_set.append(np.load(os.path.join(base_path,'t_error_%d.npy'%i)))
    r_error_set.append(np.load(os.path.join(base_path, 'angle_error_%d.npy' % i)))

t_error_set=np.concatenate(t_error_set,axis=0)
r_error_set=np.concatenate(r_error_set,axis=0)



print('total number',r_error_set.shape[0])


print(np.max(t_error_set),np.max(r_error_set))

plt.figure(1)
plt.hist(t_error_set,bins=np.arange(0,15,0.5),weights=np.ones(t_error_set.shape[0]) / t_error_set.shape[0])
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title('hist RTE')

plt.figure(2)
plt.hist(r_error_set,bins=np.arange(0,30,1),weights=np.ones(t_error_set.shape[0]) / t_error_set.shape[0])
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title('hist RRE')

plt.show()

print('RTE %0.4f +- %0.4f'%(np.mean(t_error_set),np.std(t_error_set)))
print('RRE %0.4f +- %0.4f'%(np.mean(r_error_set),np.std(r_error_set)))

'''index=(r_error_set<30)&(t_error_set<15)
print('RTE',np.mean(t_error_set[index]))
print(np.std(t_error_set[index]))
print('RRE',np.mean(r_error_set[index]))
print(np.std(r_error_set[index]))
'''
bad_index=np.where((r_error_set>30)|(t_error_set>15))[0]

good_index=np.where((r_error_set<5)&(t_error_set<2))[0]
good_rate=np.sum((r_error_set<5)&(t_error_set<2))/np.shape(r_error_set)[0]
print('successful rate %0.4f'%good_rate)
bad_t_error_set=t_error_set[bad_index]
bad_r_error_set=r_error_set[bad_index]


for i in range(np.shape(bad_t_error_set)[0]):
    print(bad_index[i],bad_t_error_set[i],bad_r_error_set[i])

