import numpy as np
# import open3d as o3d
import trimesh
from pdb import set_trace as bp

from tqdm import tqdm
import pandas as pd
import os

label_dict = {
    'yellow_anaconda': 2,
    'western_lowland_gorilla_male':1,
    'western_lowland_gorilla_female':1,
    'western_lowland_gorilla_juvenile':1,
    'western_diamondback_rattlesnake':0,
    'western_chimpanzee_male':1,
    'western_chimpanzee_female':1,
    'western_chimpanzee_juvenile':1,
    'west_african_lion_male':0,
    'west_african_lion_female':0,
    'west_african_lion_juvenile':0,
    'titan_beetle':2,
    'thomsons_gazelle_male':0,
    'thomsons_gazelle_female':0,
    'thomsons_gazelle_juvenile':0,
    'sun_bear_male':0,
    'sun_bear_female':0,
    'sun_bear_juvenile':0,
    'springbok_male':0,
    'springbok_female':0,
    'springbok_juvenile':0,
    'spotted_hyena_male':0,
    'spotted_hyena_female':0,
    'spotted_hyena_juvenile':0,
    'snow_leopard_male':0,
    'snow_leopard_female':0,
    'snow_leopard_juvenile':0,
    'siberian_tiger_male':0,
    'siberian_tiger_female':0,
    'siberian_tiger_juvenile':0,
    'saltwater_crocodile_male':2,
    'saltwater_crocodile_female':2,
    'saltwater_crocodile_juvenile':2,
    'sable_antelope_male':0,
    'sable_antelope_female':0,
    'sable_antelope_juvenile':0,
    'ring_tailed_lemur_juvenile':1,
    'ring_tailed_lemur_male':1,
    'reticulated_giraffe_male':0,
    'reticulated_giraffe_female':0,
    'reticulated_giraffe_juvenile':0,
    'reindeer_male':0,
    'reindeer_female':0,
    'reindeer_juvenile':0,
    'red_ruffed_lemur_male':1,
    'red_ruffed_lemur_female':1,
    'red_ruffed_lemur_juvenile':1,
    'red_panda_juvenile':0,
    'red_panda_male':0,
    'red_kangaroo_male':1,
    'red_kangaroo_female':1,
    'red_kangaroo_juvenile':1,
    'red_eyed_tree_frog':2,
    'pygmy_hippo_male':0,
    'pygmy_hippo_female':0,
    'pygmy_hippo_juvenile':0,
    'puff_adder':2,
    'pronghorn_antelope_male':0,
    'pronghorn_antelope_female':0,
    'pronghorn_antelope_juvenile':0,
    'proboscis_monkey_male':1,
    'proboscis_monkey_female':1,
    'proboscis_monkey_juvenile':1,
    'polar_bear_male':0,
    'polar_bear_female':0,
    'polar_bear_juvenile':0,
    'plains_zebra_male':0,
    'plains_zebra_female':0,
    'plains_zebra_juvenile':0,
    'okapi_male':1,
    'okapi_female':1,
    'okapi_juvenile':1,
    'nyala_male':0,
    'nyala_female':0,
    'nyala_juvenile':0,
    'nile_monitor_juvenile':2,
    'nile_monitor_male':2,
    'mexican_redknee_tarantula':2,
    'mandrill_male':0,
    'mandrill_female':0,
    'mandrill_juvenile':0,
    'malayan_tapir_male':0,
    'malayan_tapir_female':0,
    'malayan_tapir_juvenile':0,
    'llama_male':0,
    'llama_female':0,
    'llama_juvenile':0,
    'lesser_antillean_iguana':2,
    'lehmanns_poison_frog':2,
    ##### dragon is xiyi
    'komodo_dragon_male':2,
    'komodo_dragon_female':2,
    'komodo_dragon_juvenile':2,
    'koala_male':0,
    'koala_female':0,
    'koala_juvenile':0,
    'king_penguin_male':1,
    'king_penguin_female':1,
    'king_penguin_juvenile':1,
    'japanese_macaque_male':1,
    'japanese_macaque_female':1,
    'japanese_macaque_juvenile':1,
    'jaguar_male':0,
    'jaguar_female':0,
    'jaguar_juvenile':0,
    'indian_rhinoceros_male':0,
    'indian_rhinoceros_female':0,
    'indian_rhinoceros_juvenile':0,
    'indian_peafowl_male':1,
    'indian_peafowl_female':1,
    'indian_peafowl_juvenile':1,
    'indian_elephant_male':0,
    'indian_elephant_female':0,
    'indian_elephant_juvenile':0,
    'hippopotamus_male':0,
    'hippopotamus_female':0,
    'hippopotamus_juvenile':0,
    'himalayan_brown_bear_male':0,
    'himalayan_brown_bear_female':0,
    'himalayan_brown_bear_juvenile':0,
    'grizzly_bear_male':0,
    'grizzly_bear_female':0,
    'grizzly_bear_juvenile':0,
    'grey_seal_male':2,
    'grey_seal_female':2,
    'grey_seal_juvenile':2,
    'green_iguana':2,
    'greater_flamingo_male':1,
    'greater_flamingo_female':1,
    'greater_flamingo_juvenile':1,
    'gray_wolf_male':0,
    'gray_wolf_female':0,
    'gray_wolf_juvenile':0,
    'goliath_frog':2,
    'goliath_birdeater':2,
    'goliath_beetle':2,
    'golden_poison_frog':2,
    'gila_monster':2,
    'giant_panda_male':0,
    'giant_panda_female':0,
    'giant_panda_juvenile':0,
    'giant_otter_male':0,
    'giant_otter_female':0,
    'giant_otter_juvenile':0,
    'giant_leaf_insect':2,
    'giant_forest_scorpion':2,
    'giant_desert_hairy_scorpion':2,
    'giant_burrowing_cockroach':2,
    'giant_anteater_male':0,
    'giant_anteater_female':0,
    'giant_anteater_juvenile':0,
    'gharial_male':2,
    'gharial_female':2,
    'gharial_juvenile':2,
    'gemsbok_male':0,
    'gemsbok_female':0,
    'gemsbok_juvenile':0,
    ##### it is turtle
    'galapagos_giant_tortoise_male':2,
    'galapagos_giant_tortoise_female':2,
    'galapagos_giant_tortoise_juvenile':2,

    'formosan_black_bear_male':0,
    'formosan_black_bear_female':0,
    'formosan_black_bear_juvenile':0,

    'eastern_brown_snake':2,
    'eastern_blue_tongued_lizard':2,
    'dingo_male':0,
    'dingo_female':0,
    'dingo_juvenile':0,
    'diamondback_terrapin':2,
    'dhole_male':0,
    'dhole_female':0,
    'dhole_juvenile':0,
    'death_adder':2,
    'dall_sheep_male':0,
    'dall_sheep_female':0,
    'dall_sheep_juvenile':0,
    'cuviers_dwarf_caiman_male':2,
    'cuviers_dwarf_caiman_female':2,
    'cuviers_dwarf_caiman_juvenile':2,
    'common_warthog_male':0,
    'common_warthog_female':0,
    'common_warthog_juvenile':0,
    'common_ostrich_male':1,
    'common_ostrich_female':1,
    'common_ostrich_juvenile':1,
    'clouded_leopard_male':0,
    'clouded_leopard_female':0,
    'clouded_leopard_juvenile':0,
    'chinese_pangolin_male':0,
    'chinese_pangolin_female':0,
    'chinese_pangolin_juvenile':0,
    'cheetah_male':0,
    'cheetah_female':0,
    'cheetah_juvenile':0,
    'cassowary_male':1,
    'cassowary_female':1,
    'cassowary_juvenile':1,
    'capuchin_monkey_male':1,
    'capuchin_monkey_female':1,
    'capuchin_monkey_juvenile':1,
    'brazilian_wandering_spider':2,
    'brazilian_salmon_pink_tarantula':2,
    'bornean_orangutan_male':1,
    'bornean_orangutan_female':1,
    'bornean_orangutan_juvenile':1,
    'bonobo_male':1,
    'bonobo_female':1,
    'bonobo_juvenile':1,
    'bongo_male':0,
    'bongo_female':0,
    'bongo_juvenile':0,
    'boa_constrictor':2,
    'black_wildebeest_male':0,
    'black_wildebeest_female':0,
    'black_wildebeest_juvenile':0,
    'binturong_male':0,
    'binturong_female':0,
    'binturong_juvenile':0,
    'bengal_tiger_male':0,
    'bengal_tiger_female':0,
    'bengal_tiger_juvenile':0,
    'bairds_tapir_male':0,
    'bairds_tapir_female':0,
    'bairds_tapir_juvenile':0,
    'bactrian_camel_male':0,
    'bactrian_camel_female':0,
    'bactrian_camel_juvenile':0,
    'babirusa_male':0,
    'babirusa_female':0,
    'babirusa_juvenile':0,
    'arctic_wolf_male':0,
    'arctic_wolf_female':0,
    'arctic_wolf_juvenile':0,
    'american_bison_male':0,
    'american_bison_female':0,
    'american_bison_juvenile':0,
    'amazonian_giant_centipede':2,
    'aldabra_giant_tortoise_male':2,
    'aldabra_giant_tortoise_female':2,
    'aldabra_giant_tortoise_juvenile':2,
    'african_wild_dog_male':0,
    'african_wild_dog_female':0,
    'african_wild_dog_juvenile':0,
    'african_elephant_male':0,
    'african_elephant_female':0,
    'african_elephant_juvenile':0,
    'african_buffalo_male':0,
    'african_buffalo_female':0,
    'african_buffalo_juvenile':0,
    'aardvark_male':0,
    'aardvark_female':0,
    'aardvark_juvenile':0

}

def create_grid(resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1]), transform=None):
    '''
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    '''
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    return coords, coords_matrix


def eval_grid(coords, eval_func, num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]
    coords = coords.reshape([3, -1])
    # bp()
    sdf = batch_eval_new(coords, eval_func, num_samples=num_samples)
    # BP()
    # bp()
    gt = np.where(sdf==1)[0].shape[0]
    inter = np.where(sdf==0)[0].shape[0]
    recon = np.where(sdf==-1)[0].shape[0]

    print("GT ",gt)
    print("Inter ",inter)
    print("Recon ",recon)
    score = (inter-gt-recon)/(inter+gt)
    print('Score is ',score)
    result_dic = {'GT':gt,'Inter':inter,'Recon':recon,'Score':score}
    return result_dic

def batch_eval_new(points, eval_func, num_samples=512 * 512 * 512,flag=None):
    num_pts = points.shape[1]
    sdf = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in tqdm(range(num_batches)):
        sdf[i * num_samples:i * num_samples + num_samples] = eval_func(
            points[:, i * num_samples:i * num_samples + num_samples])
        # bp()
    if num_pts % num_samples:
        sdf[num_batches * num_samples:] = eval_func(points[:, num_batches * num_samples:])

    return sdf


def eval_sdf(coords, eval_func,
                     init_resolution=64, threshold=0.01,
                     num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]

    sdf = np.zeros(resolution)

    # dirty = np.ones(resolution, dtype=np.bool)
    # grid_mask = np.zeros(resolution, dtype=np.bool)
    dirty = np.ones(resolution)
    grid_mask = np.zeros(resolution)

    reso = resolution[0] // init_resolution

    # bp()
    while reso > 0:
        print("Reso ",reso)
        # subdivide the grid
        grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso, 0:resolution[2]:reso] = True
        # test samples in this iteration
        test_mask = np.logical_and(grid_mask, dirty)
        #print('step size:', reso, 'test sample size:', test_mask.sum())
        points = coords[:, test_mask]

        sdf[test_mask] = batch_eval_new(points, eval_func, num_samples=num_samples)
        dirty[test_mask] = False

        # do interpolation
        if reso <= 1:
            break
        for x in range(0, resolution[0] - reso, reso):
            for y in range(0, resolution[1] - reso, reso):
                for z in range(0, resolution[2] - reso, reso):
                    # if center marked, return
                    if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                        continue
                    v0 = sdf[x, y, z]
                    v1 = sdf[x, y, z + reso]
                    v2 = sdf[x, y + reso, z]
                    v3 = sdf[x, y + reso, z + reso]
                    v4 = sdf[x + reso, y, z]
                    v5 = sdf[x + reso, y, z + reso]
                    v6 = sdf[x + reso, y + reso, z]
                    v7 = sdf[x + reso, y + reso, z + reso]
                    v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                    v_min = v.min()
                    v_max = v.max()
                    # this cell is all the same
                    if (v_max - v_min) < threshold:
                        sdf[x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
                        dirty[x:x + reso, y:y + reso, z:z + reso] = False
        reso //= 2

    return sdf.reshape(resolution)


def compute_metric(resolution, b_min, b_max,mesh_gt,mesh_tar,
                   use_octree=False, num_samples=200000, transform=None, cam_loc=None, root=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)

#     mesh = trimesh.load('/media/ivenwu/My_Disk/video_obj_sample_static/african_elephant_female/frame_000001.obj')
    # Then we define the lambda function for cell evaluation
    

    # bp()
    def eval_func(points):
        # print("Evaluating")
        
        points = points.T
        # pred_gt = mesh_gt.contains(points).astype(np.int32)
        pred_gt = mesh_gt.contains(points)
        # pred_gt = mesh_tar.contains(points)
        pred_tar = mesh_tar.contains(points)

        intersection = np.logical_and(pred_gt,pred_tar).astype(np.int32)
        

        pred = np.ones(pred_gt.shape)*200
        pred[np.where(pred_gt==1)] = 1
        pred[np.where(pred_tar==1)] = -1
        pred[np.where(intersection==1)] = 0

#         bp()
        return pred


    # Then we evaluate the grid

    result = eval_grid(coords,eval_func,num_samples=num_samples)

    return result
    # bp()

if __name__ =='__main__':

    
    res = pd.DataFrame(columns=('animal1', 'scale1','species1','animal2','scale2', 'species2','Score','GT','Recon','Inter'))

    res.to_csv('result.csv')
    mesh_ori = '/media/ivenwu/My_Disk/video_obj_join/'

    
    animal_list = os.listdir(mesh_ori)

    # for i in animal_list:
    #     if i not in label_dict.keys():
    #         print(i)
    
    # return

    for i in range(len(animal_list)):
        for j in range(i,len(animal_list)):

            # animal1 = 'african_elephant_female'
            # animal2 = 'bairds_tapir_juvenile'
            animal1 = animal_list[i]
            animal2 = animal_list[j]
            print('animal1: ',animal1)
            print('animal2: ',animal2)

            mesh_gt = trimesh.load('/media/ivenwu/My_Disk/video_obj_join/{}/frame_000001.obj'.format(animal1))
            mesh_tar = trimesh.load('/media/ivenwu/My_Disk/video_obj_join/{}/frame_000001.obj'.format(animal2))
            tmp = {'animal1':animal1,'scale1':mesh_gt.extents,'species':label_dict[animal1],'animal2':animal2,'scale2':mesh_tar.extents,'species2':label_dict[animal2]}
            result = compute_metric(resolution=256,b_min=np.array([-2,-2,-2]),b_max=np.array([4,4,4]),mesh_gt=mesh_gt,mesh_tar=mesh_tar)

            tmp.update(
                {
                    'Score':result['Score'],
                    'GT':result['GT'],
                    'Recon':result['Recon'],
                    'Inter':result['Inter']
                }
            )

            res = res.append([tmp],ignore_index=True)
            res.to_csv('result.csv',mode='a',header=False)
    # res.to_csv('result.csv')
    # with open('result.txt','w') as f:
#     f.write(animal1+' '+animal2+' '+str(result['Score']))