import os
from tqdm import tqdm

tex_dir = '/scratch/users/yuefanw/planetzoo/'
file_dir = '/scratch/users/yuefanw/dataset/custom_PIFu_init_data/'

name_list = []
for animal in tqdm(os.listdir(file_dir)):
    # name_list.append(animal)
    flag = 0
    for single_path in os.walk(tex_dir):
        i, j, k = single_path
        for t in k:
            if 'masktexture.PNG' in t:
                filepath = i + '/' + t
                command = 'cp {} {}'.format(filepath, os.path.join(file_dir, animal, t))
                os.system(command)
                flag = 1
                break

        if flag == 1:
            break


