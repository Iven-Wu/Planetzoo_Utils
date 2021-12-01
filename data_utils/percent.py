import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as bp

bias = np.load('/home/ivenwu/下载/RAFT-3D-master/bias.npy')

max_value = 100
min_value = 0

split_length = 5

count_array = np.zeros(int(100/split_length))

name_list = [i*split_length for i in range(int(100/split_length))]

pp_list = []
for value in bias:
    index = int(value//split_length)
    # bp(s)
    count_array[index] += 1

    # pp_list.append()

plt.bar(range(len(count_array)), count_array,fc='r',tick_label=name_list)

plt.show()
