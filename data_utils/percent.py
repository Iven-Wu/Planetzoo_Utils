import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as bp

bias = np.load('/home/ivenwu/下载/RAFT-3D-master/bias.npy')

max_value = 100
min_value = 0

count_array = np.zeros(10)

name_list = [i*10 for i in range(10)]
for value in bias:
    index = int(value//10)
    # bp(s)
    count_array[index] += 1

plt.bar(range(len(count_array)), count_array,fc='r',tick_label=name_list)

plt.show()
