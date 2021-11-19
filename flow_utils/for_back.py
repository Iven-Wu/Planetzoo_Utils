import sys
sys.path.append('.')

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import torch
import torch.nn.functional as F

from PIL import Image
from lietorch import SE3
import raft3d.projective_ops as pops
from data_readers import frame_utils
from utils import show_image, normalize_image
from pdb import set_trace as bp

from tqdm import tqdm

import open3d as o3d

DEPTH_SCALE = 0.2
# DEPTH_SCALE = 0.1
def prepare_images_and_depths(image1, image2, depth1, depth2):
    """ padding, normalization, and scaling """

    # bp()
    # image1 = F.pad(image1, [0,0,0,4], mode='replicate')
    # image2 = F.pad(image2, [0,0,0,4], mode='replicate')
    # depth1 = F.pad(depth1[:,None], [0,0,0,4], mode='replicate')[:,0]
    # depth2 = F.pad(depth2[:,None], [0,0,0,4], mode='replicate')[:,0]

    depth1 = (DEPTH_SCALE * depth1).float()
    depth2 = (DEPTH_SCALE * depth2).float()
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)

    return image1, image2, depth1, depth2


def display(img, tau, phi):
    """ display se3 fields """
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(img[:, :, :] / 255.0)

    tau_img = np.clip(tau, -0.1, 0.1)
    tau_img = (tau_img + 0.1) / 0.2

    phi_img = np.clip(phi, -0.1, 0.1)
    phi_img = (phi_img + 0.1) / 0.2

    ax2.imshow(tau_img)
    ax3.imshow(phi_img)
    # ax2.imshow(tau/255.0)
    # ax3.imshow(phi/255.0)

    plt.show()

def down_resolution(img):
    tmp = Image.fromarray(img)
    tmp = tmp.resize((960,540))
    res = np.array(tmp)
    return res

@torch.no_grad()
def demo(args,index1=1,index2=2,want_display=True):
    import importlib
    RAFT3D = importlib.import_module(args.network).RAFT3D
    model = torch.nn.DataParallel(RAFT3D(args))
    model.load_state_dict(torch.load(args.model), strict=False)

    model.eval()
    model.cuda()

    # fx, fy, cx, cy = (1050.0, 1050.0, 480.0, 270.0)
    # fx,fy,cx,cy = (4533.3/2,45333.3/2,959.5/2,539.5/2)

    intrin = np.load(os.path.join(args.root,'info','0001.npz'))['intrinsic_mat']
    fx,fy,cx,cy = intrin[0,0],intrin[1,1],intrin[-1,0],intrin[-1,1]
    # fx,fy,cx,cy = (2266.7,2266.7,479.5,269.5)

    # img1 = cv2.imread('assets/image1.png')
    # img2 = cv2.imread('assets/image2.png')
    # disp1 = frame_utils.read_gen('assets/disp1.pfm')
    # disp2 = frame_utils.read_gen('assets/disp2.pfm')

    # # bp()
    # img1 = cv2.imread('assets/0003.png')
    # img2 = cv2.imread('assets/0004.png')

    # index1 = 1
    # index2 = 2
    img1 = cv2.imread(os.path.join(args.root,'{:04d}.png'.format(index1)))
    img2 = cv2.imread(os.path.join(args.root,'{:04d}.png'.format(index2)))

    # bp()
    # disp1 = frame_utils.read_gen('')

    # disp1 = np.load('assets/0003.npz')['depth_map']
    # disp2 = np.load('assets/0004.npz')['depth_map']
    disp1 = np.load(os.path.join(args.root,'info','{:04d}.npz'.format(index1)))['depth_map']
    disp2 = np.load(os.path.join(args.root,'info','{:04d}.npz'.format(index2)))['depth_map']

    # img1,img2,disp1,disp2 = down_resolution(img1),down_resolution(img2),down_resolution(disp1),down_resolution(disp2)
    
    # bp()
    # bp()

    # depth1 = torch.from_numpy(fx / disp1).float().cuda().unsqueeze(0)
    # depth2 = torch.from_numpy(fx / disp2).float().cuda().unsqueeze(0)
    depth1 = torch.from_numpy(disp1).float().cuda().unsqueeze(0)
    depth2 = torch.from_numpy(disp2).float().cuda().unsqueeze(0)
    image1 = torch.from_numpy(img1).permute(2,0,1).float().cuda().unsqueeze(0)
    image2 = torch.from_numpy(img2).permute(2,0,1).float().cuda().unsqueeze(0)
    intrinsics = torch.as_tensor([fx, fy, cx, cy]).cuda().unsqueeze(0)

    # bp()
    image1, image2, depth1, depth2 = prepare_images_and_depths(image1, image2, depth1, depth2)

    # bp()
    Ts = model(image1, image2, depth1, depth2, intrinsics, iters=16)
    
    # compute 2d and 3d from from SE3 field (Ts)
    flow2d, flow3d, _ = pops.induced_flow(Ts, depth1, intrinsics)

    # extract rotational and translational components of Ts

    # bp()
    tau, phi = Ts.log().split([3,3], dim=-1)
    tau = tau[0].cpu().numpy()
    phi = phi[0].cpu().numpy()

    # undo depth scaling
    flow3d = flow3d / DEPTH_SCALE
    # bp()
    
    # fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    # bp()

    # bp()

    # display(img1, img2, img1+flow3d[0].cpu().numpy())
    if want_display:
        display(img1,tau,phi)

    # bp()
    # Image_img = Image.fromarray(img1)
    # Image_img.save('img1.png')

    # tau_img = Image.fromarray(tau[:-4])
    # tau_img.save('tau.png')

    # phi_img = Image.fromarray(phi[:-4])
    # phi_img.save('phi.png')
    # display(flow3d[0,:,:,0],flow3d[0,:,:,1],flow3d[0,:,:,-1])

    # bp()
    # np.save('aaa.npy',flow3d[0].cpu().numpy())
    np.save('./flow/{}to{}.npy'.format(index1,index2),flow3d[0].detach().cpu().numpy())

    return flow3d


def get_recover(info):

    intrin = np.eye(4)
    intrin[:-1,:-1] = info['intrinsic_mat']
    extrin1 = info['extrinsic_mat']
    extrin = np.concatenate([extrin1,np.array([0,0,0,1]).reshape(1,-1)],axis=0)

    recover_mat = np.linalg.pinv(intrin.dot(extrin))

    return recover_mat

def get_3d(info):

    # bp()
    recover = get_recover(info)
    depth = info['depth_map']
    seg = info['segmentation_masks']
    x,y = np.where(seg!=0)
    select_co = co = np.concatenate([x.reshape(1,-1),y.reshape(1,-1)]).T
    z = [depth[c[0],c[1]] for c in select_co]
    z = np.array(z).reshape(-1,1)
    xyz = np.concatenate([select_co*z,z],axis=1)
    xyz = xyz[:,[1,0,2]]    
    xyzw = np.concatenate([xyz,np.ones((xyz.shape[0],1))],axis=1)
    recon = recover.dot(xyzw.T).T
    recon = recon[:,:-1]
    recon[:,1] *= -1
    recon[:,-1] *= -1

    return recon,select_co

def get_proj(info,points):
    intrin = info['intrinsic_mat']
    extrin = info['extrinsic_mat']

    calib = intrin.dot(extrin)

    # points = np.array(mesh.vertices)
    # points = points[:,[0,2,1]]
    # points[:,1]*= -1
    ones_array = np.ones(points.shape[0])
    homo_points = np.concatenate([points,ones_array.reshape(-1,1)],axis=1)

    project = calib.dot(homo_points.T).T

    project =  project/project[:,[2]]
    return project

def forward_and_backward(args,index1=1,index2=2):
    flow_for = demo(args,index1=index1,index2=index2,want_display=False)
    flow_for = flow_for[0]
    info1 = np.load(os.path.join(args.root,'info','0001.npz'))
    
    recon1,select_co1 = get_3d(info1)

    init_proj = get_proj(info1,recon1)

    select_flow1 = np.array([flow_for[select_single_co[0],select_single_co[1]].cpu().numpy() for select_single_co in select_co1])
    select_flow1 = select_flow1[:,[1,0,2]]
    select_flow1[:,0] *= -1

    change1 = recon1 + select_flow1 


    
    # bp()

    ##################################
    flow_back = demo(args,index1=index2,index2=index1,want_display=False)
    flow_back = flow_back[0]
    info2 = np.load(os.path.join(args.root,'info','0002.npz'))


    proj_chg = get_proj(info2,change1)
    proj_chg = np.rint(proj_chg).astype(np.int)[:,:-1]

    recon2,select_co2 = get_3d(info2)
    # bp()


    select_flow2 = np.array([flow_back[select_single_co[0],select_single_co[1]].cpu().numpy() for select_single_co in proj_chg])

    # select_flow2 = select_flow2[:,[1,0,2]]
    # select_flow2[:,0] *= -1

    #####################
    for_and_back = recon1+select_flow1+select_flow2

    new_proj = get_proj(info1,for_and_back)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(18,12))
    plt.plot(new_proj[:, 0], new_proj[:, 1], '.')
    plt.show()

    bias = new_proj-init_proj
    # bias = recon1-for_and_back
    bp()

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='raft3d.pth', help='checkpoint to restore')
    parser.add_argument('--network', default='raft3d.raft3d', help='network architecture')
    parser.add_argument('--root',default='/media/ivenwu/My_Disk/video_join/african_elephant_female/')
    args = parser.parse_args()


    # forward_and_backward(args)

    demo(args,index1=2,index2=1)
    # for i in tqdm(range(1,100)):
    #     demo(args,index1=i,index2=10)
    #     break

    

    



