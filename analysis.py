import numpy as np
import os
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

train_safety_radius=1
test_safety_radius=1

pos_margin=0.2
neg_margin=1.8
print('train_safety_radius ',train_safety_radius)
print('test_safety_radius ',test_safety_radius)

#base_path='H:\\e\\DenseI2P_exp\\2022-01-17\\result_error_%0.2f_%0.2f_pos_margin_%0.2f_neg_margin_%0.2f'%(train_safety_radius,test_safety_radius,pos_margin,neg_margin)
#base_path='D:\\DenseI2P_exp\\2022-01-17\\3DROT\\result_error_%0.2f_%0.2f_pos_margin_%0.2f_neg_margin_%0.2f'%(train_safety_radius,test_safety_radius,pos_margin,neg_margin)
#base_path='E:\\DenseI2P_exp\\2021-11-23\\result_error'

base_path='.'
base_path=os.path.join(base_path,'result_error_%0.2f_%0.2f_pos_margin_%0.2f_neg_margin_%0.2f'%(train_safety_radius,test_safety_radius,pos_margin,neg_margin))

total_num=len(os.listdir(base_path))//5

t_error_set=[]
r_error_set=[]

for i in range(total_num):
    t_error_set.append(np.load(os.path.join(base_path,'t_error_%d.npy'%i)))
    r_error_set.append(np.load(os.path.join(base_path, 'angle_error_%d.npy' % i)))

t_error_set=np.concatenate(t_error_set,axis=0)
r_error_set=np.concatenate(r_error_set,axis=0)

'''index=r_error_set<100
t_error_set=t_error_set[index]
r_error_set=r_error_set[index]'''
#t_error_set[t_error_set>=14]=14

print('total number',r_error_set.shape[0])

print(np.max(t_error_set),np.max(r_error_set))
print(np.argmax(t_error_set),np.argmax(r_error_set))

plt.figure(1)
plt.hist(t_error_set,bins=np.arange(0,15,0.5),weights=np.ones(t_error_set.shape[0]) / t_error_set.shape[0])
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title('hist RTE')
plt.savefig('RTE.png')

plt.figure(2)
plt.hist(r_error_set,bins=np.arange(0,30,1),weights=np.ones(t_error_set.shape[0]) / t_error_set.shape[0])
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title('hist RRE')
plt.savefig('RRE.png')

plt.figure(3)
RTE_range=np.arange(0,15,0.5)
RTE_ratio=[]
for i in RTE_range:
    RTE_ratio.append(np.sum(t_error_set<i)/np.shape(t_error_set)[0])
RTE_ratio=np.array(RTE_ratio)
plt.plot(RTE_range,RTE_ratio)
plt.savefig('RTE_ratio.png')

plt.figure(4)
RRE_range=np.arange(0,30,1)
RRE_ratio=[]
for i in RRE_range:
    RRE_ratio.append(np.sum(r_error_set<i)/np.shape(t_error_set)[0])
RRE_ratio=np.array(RRE_ratio)
plt.plot(RRE_range,RRE_ratio)
plt.savefig('RRE_ratio.png')



#plt.show()

print('RTE %0.4f\pm%0.4f'%(np.mean(t_error_set),np.std(t_error_set)))
print('RRE %0.4f\pm%0.4f'%(np.mean(r_error_set),np.std(r_error_set)))

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

very_bad=np.where((r_error_set>100)|(t_error_set>15))[0]


'''for i in range(np.shape(very_bad)[0]):
    print(very_bad[i],t_error_set[very_bad[i]],r_error_set[very_bad[i]])
'''

'''for i in range(np.shape(good_index)[0]):
    print(good_index[i],t_error_set[good_index[i]],r_error_set[good_index[i]])
'''