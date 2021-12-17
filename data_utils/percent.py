import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as bp

bias = np.load('/home/ivenwu/下载/RAFT-3D-master/bias_new.npy')

max_value = 40
min_value = 0

split_length = 2
# bp()
count_array = np.zeros(int(max_value/split_length))

name_list = [i*split_length for i in range(int(max_value/split_length))]

pp_list = []
for value in bias[:,0]:
    index = int(abs(value)//split_length)
    # bp(s)
    count_array[index] += 1

    # pp_list.append()

plt.bar(range(len(count_array)), count_array,fc='r',tick_label=name_list)

plt.show()

count_array = np.zeros(int(max_value/split_length))

for value in bias[:,1]:
    index = int(abs(value)//split_length)
    # bp(s)
    count_array[index] += 1

    # pp_list.append()

plt.bar(range(len(count_array)), count_array,fc='r',tick_label=name_list)

plt.show()

count_array = np.zeros(int(max_value/split_length))

for value in bias[:,2]:
    index = int(abs(value)//split_length)
    # bp(s)
    count_array[index] += 1

    # pp_list.append()

plt.bar(range(len(count_array)), count_array,fc='r',tick_label=name_list)

plt.show()