import open3d as o3d
import numpy as np
import os

def downsize(mesh_in,target_face_num=7500):
    if type(mesh_in)==str:
        mesh_in = o3d.io.read_triangle_mesh(mesh_in)
    new = mesh_in.simplify_quadric_decimation(target_number_of_triangles=target_face_num)
    
    return new
    
   
file_folder = 'reconstruction1'

for mesh_name in os.listdir(file_folder):
	mesh = o3d.io.read_triangle_mesh(os.path.join(file_folder,mesh_name))
	new_mesh = downsize(mesh)
	o3d.io.write_triangle_mesh(os.path.join(file_folder,'{}.obj'.format(mesh_name[:-4])), new_mesh) 
	o3d.io.write_triangle_mesh(os.path.join(file_folder,'{}_remesh.obj'.format(mesh_name[:-4])), new_mesh) 
