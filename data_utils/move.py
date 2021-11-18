import os
from tqdm import tqdm
ori_dir = '/home/yuefanw/scratch/planetzoo_obj/'
# out_dir = '/home/yuefanw/scratch/dataset/custom_PIFu_init_data/'
out_dir = '/home/yuefanw/scratch/dataset/metric_data/'
for animal_folder in tqdm(os.listdir(ori_dir)):
    if not os.path.exists(os.path.join(out_dir,animal_folder)):
        os.makedirs(os.path.join(out_dir,animal_folder))
    if not os.path.exists(os.path.join(ori_dir,animal_folder,'frame_000001.obj')):
        print(animal_folder)
    command  = 'cp {} {}'.format(os.path.join(ori_dir,animal_folder,'frame_000001.obj'), os.path.join(out_dir,animal_folder,'frame_000001.obj'))
    os.system(command)

# for animal_folder in tqdm(os.listdir(ori_dir)):

