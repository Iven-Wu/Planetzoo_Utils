import os
from tqdm import tqdm
def delete_thing(file_name):
    with open(file_name,'r') as r:
        lines = r.readlines()
    with open(file_name,'w') as w:
        for l in lines:
            if 'usemtl' not in l:
                w.write(l)

if __name__ =='__main__':
    folder_dir = '/scratch/users/yuefanw/version1/'
    for folder in tqdm(os.listdir(folder_dir)):
        for single_file in os.listdir(os.path.join(folder_dir,folder)):
            if '.obj' in single_file:
                delete_thing(os.path.join(folder_dir,folder,single_file))
