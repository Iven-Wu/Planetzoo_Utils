import open3d as o3d
import numpy as np
import os

def downsize(mesh_in,target_face_num=6500):
    if type(mesh_in)==str:
        mesh_in = o3d.io.read_triangle_mesh(mesh_in)
    new = mesh_in.simplify_quadric_decimation(target_number_of_triangles=6500)
    
    return new
    
    
   

folder_dir = 'reconstruction_object'

for obj_name in os.listdir(folder_dir):
    mesh = o3d.io.read_triangle_mesh(os.path.join(folder_dir,obj_name))
    new_mesh = downsize(mesh)
    o3d.io.write_triangle_mesh(os.path.join(folder_dir,'{}_remesh.obj'.format(obj_name[:-4])),new_mesh)
