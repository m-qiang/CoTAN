import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def apply_affine(v, affine):
    """apply affine transform to the surface"""
    v_tmp = np.ones([v.shape[0],4])
    v_tmp[:,:3] = v
    return v_tmp.dot(affine.T)[:,:3]


def adjacency(f):
    """Compute the adjacency matrix given the mesh faces"""
    with torch.no_grad():
        nv = f.max().item()+1
        e = torch.cat([f[0,:,[0,1]],
                       f[0,:,[1,2]],
                       f[0,:,[2,0]]], dim=0).T
        # adjacency matrix
        adj_matrix = torch.sparse_coo_tensor(e, torch.ones_like(e[0]).float(),
                                    (nv, nv)).unsqueeze(0)
        # number of neighbors for each vertex
        adj_degree = torch.sparse.sum(adj_matrix, dim=-1).to_dense().unsqueeze(-1)
    return adj_matrix, adj_degree


def edge_to_face(f):
    """find the adjacent two faces of each edge"""
    edge = torch.cat([f[0,:,[0,1]],
                      f[0,:,[1,2]],
                      f[0,:,[2,0]]], axis=0)  # (2|E|, 2)
    nf = f.shape[1]
    # map the edge to its belonging face
    fid = torch.arange(nf).to(f.device)
    e2f = torch.cat([fid]*3)  # (2|E|, 2)

    edge = edge.cpu().numpy()
    # sort the edge such that v_i < v_j
    edge = np.sort(edge, axis=-1)
    # sort the edge to find the correspondence 
    # between e_ij and e_ji
    eid = np.lexsort((edge[:,1], edge[:,0]))  # (2|E|)

    # map edge to its adjacent two faces
    e2f = e2f[eid].reshape(-1,2)  # (|E|, 2)
    return e2f


def compute_vert_normal(v, f):
    """
    Compute the normal of each vertex based on pytorch3d.structures.meshes.
    For original code please see _compute_vertex_normals function in:
    - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/structures/meshes.html    
    """
    
    n_v = torch.zeros_like(v)   # normals of vertices
    v_f = v[:, f[0]]

    # compute normals of faces
    n_f_0 = torch.cross(v_f[:,:,1]-v_f[:,:,0], v_f[:,:,2]-v_f[:,:,0], dim=2) 
    n_f_1 = torch.cross(v_f[:,:,2]-v_f[:,:,1], v_f[:,:,0]-v_f[:,:,1], dim=2) 
    n_f_2 = torch.cross(v_f[:,:,0]-v_f[:,:,2], v_f[:,:,1]-v_f[:,:,2], dim=2) 

    # sum the faces normals
    n_v = n_v.index_add(1, f[0,:,0], n_f_0)
    n_v = n_v.index_add(1, f[0,:,1], n_f_1)
    n_v = n_v.index_add(1, f[0,:,2], n_f_2)

    n_v = n_v / torch.norm(n_v, dim=-1).unsqueeze(-1) #  + 1e-12)
        
    return n_v


def compute_face_normal(v, f):
    """
    compute the normal of each face
    """
    v_f = v[:, f[0]]
    # compute normals of faces
    n_f = torch.cross(v_f[:,:,1]-v_f[:,:,0], v_f[:,:,2]-v_f[:,:,0], dim=2) 
    n_f = n_f / (torch.norm(n_f, dim=-1).unsqueeze(-1) + 1e-12)
    return n_f


from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance

point_face_distance = _PointFaceDistance.apply

def point_to_mesh_dist(pcls, meshes):
    """
    Compute point to mesh distance based on pytorch3d.loss.point_mesh_face_distance.
    For original code please see:
    - https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/point_mesh_distance.html
    """
    
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    return point_to_face.sqrt()


def compute_mesh_distance(
    v_pred, v_gt, f_pred, f_gt, n_pts=100000, seed=10086):
    """ Compute average symmetric surface distance (ASSD) and Hausdorff distance (HD). """
    
    mesh_pred = Meshes(verts=list(v_pred), faces=list(f_pred))
    mesh_gt = Meshes(verts=list(v_gt), faces=list(f_gt))
    pts_pred = sample_points_from_meshes(mesh_pred, num_samples=n_pts)
    pts_gt = sample_points_from_meshes(mesh_gt, num_samples=n_pts)
    pcl_pred = Pointclouds(pts_pred)
    pcl_gt = Pointclouds(pts_gt)

    x_dist = point_to_mesh_dist(pcl_pred, mesh_gt)
    y_dist = point_to_mesh_dist(pcl_gt, mesh_pred)

    assd = (x_dist.mean().item() + y_dist.mean().item()) / 2

    x_quantile = torch.quantile(x_dist, 0.9).item()
    y_quantile = torch.quantile(y_dist, 0.9).item()
    hd = max(x_quantile, y_quantile)
    
    return assd, hd



import nibabel as nib
from nibabel.gifti import gifti

def save_gifti_surface(v, f, save_path,
                       surf_hemi='CortexLeft',
                       surf_type='GrayWhite',
                       geom_type='Anatomical'):

    """
    - surf_hemi: ['CortexLeft', 'CortexRight']
    - surf_type: ['GrayWhite', 'Pial', 'MidThickness']
    - geom_type: ['Anatomical', 'VeryInflated', 'Spherical', 'Inflated']
    """
    v = v.astype(np.float32)
    f = f.astype(np.int32)

    # meta data
    v_meta_dict = {'AnatomicalStructurePrimary': surf_hemi,
                   'AnatomicalStructureSecondary': surf_type,
                   'GeometricType': geom_type,
                   'Name': '#1'}
    f_meta_dict = {'Name': '#2'}

    v_meta = gifti.GiftiMetaData()
    f_meta = gifti.GiftiMetaData()
    v_meta = v_meta.from_dict(v_meta_dict)
    f_meta = f_meta.from_dict(f_meta_dict)

    # new gifti image
    gii_surf = gifti.GiftiImage()

    gii_surf_v = gifti.GiftiDataArray(v, intent='pointset', meta=v_meta)
    gii_surf_f = gifti.GiftiDataArray(f, intent='triangle', meta=f_meta)
    gii_surf.add_gifti_data_array(gii_surf_v)
    gii_surf.add_gifti_data_array(gii_surf_f)

    nib.save(gii_surf, save_path)
