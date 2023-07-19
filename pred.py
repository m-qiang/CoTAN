import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import nibabel as nib
import argparse
import time
from cotan import CoTAN
from util import apply_affine, save_gifti_surface



if __name__ == "__main__":
    
    # ------ load arguments ------ 
    parser = argparse.ArgumentParser(description="CoTAN")
    
    parser.add_argument('--data_path', default='./dataset/', type=str, help="directory of the input")
    parser.add_argument('--model_path', default='./model/', type=str, help="directory of the saved models")
    parser.add_argument('--save_path', default='./result/', type=str, help="directory to save the surfaces")
    parser.add_argument('--data_name', default='dhcp', type=str, help="[dhcp, ...]")
    parser.add_argument('--age', default=40.0, type=float, help="age of the neonates")
    parser.add_argument('--device', default="cuda", type=str, help="cuda or cpu")
    parser.add_argument('--step_size', default=0.02, type=float, help="integration step size")
    parser.add_argument('--n_svf', default=4, type=int, help="number of velocity fields")
    parser.add_argument('--n_res', default=3, type=int, help="number of scales")

    args = parser.parse_args()
    
    data_path = args.data_path  # directory of the input mri
    model_path = args.model_path  # directory of the saved models
    save_path = args.save_path  # directory to save the surface
    data_name = args.data_name  # dhcp
    age = args.age  # age of the neonates
    device = torch.device(args.device)
    step_size = args.step_size
    M = args.n_svf
    R = args.n_res

    
    # ------- load template ------- 
    print('Load template ...')
    vol_mni = nib.load('./template/mni152_brain_clip.nii.gz')
    affine_in = vol_mni.affine
    surf_left_in = nib.load('./template/init_surf_left.surf.gii')
    surf_right_in = nib.load('./template/init_surf_right.surf.gii')
    
    v_left_in = surf_left_in.agg_data('pointset')
    f_left_in = surf_left_in.agg_data('triangle')
    v_left_in = apply_affine(v_left_in, np.linalg.inv(affine_in))
    v_left_in[:,0] = v_left_in[:,0] - 64
    v_left_in = (v_left_in - [56, 112, 80]) / 112
    f_left_in = f_left_in[:,[2,1,0]]
    v_left_in = torch.Tensor(v_left_in[None]).to(device)
    f_left_in = torch.LongTensor(f_left_in[None]).to(device)
    
    v_right_in = surf_right_in.agg_data('pointset')
    f_right_in = surf_right_in.agg_data('triangle')
    v_right_in = apply_affine(v_right_in, np.linalg.inv(affine_in))
    f_right_in = f_right_in[:,[2,1,0]]
    v_right_in = (v_right_in - [56, 112, 80]) / 112
    v_right_in = torch.Tensor(v_right_in[None]).to(device)
    f_right_in = torch.LongTensor(f_right_in[None]).to(device)
    
    # input integration time sequence
    T = torch.arange(1./step_size).to(device).unsqueeze(1) * step_size

    
    # ------ load input volume and age ------
    vol = nib.load(data_path)
    affine_t2 = vol.affine 
    vol_arr = (vol.get_fdata() / 40.).astype(np.float32)  # normalize intensity
    vol_in = torch.Tensor(vol_arr[None,None]).to(device)
    vol_left_in = vol_in[:,:,64:]
    vol_right_in = vol_in[:,:,:112]
    
    # input age
    age = (age-20) / 30.  # normalize age
    age = torch.Tensor(np.array(age)[None]).to(device)
    age = age.repeat(int(1./step_size), 1).to(device)
    
    
    # ------ initialize model ------ 
    print('Initalize model ...')
    model_left_white = CoTAN(
        layers=[16,32,64,128,128], M=M, R=R).to(device)
    model_left_pial = CoTAN(
        layers=[16,32,32,32,32], M=M, R=R).to(device)
    model_right_white = CoTAN(
        layers=[16,32,64,128,128], M=M, R=R).to(device)
    model_right_pial = CoTAN(
        layers=[16,32,32,32,32], M=M, R=R).to(device)
    
    model_left_white.load_state_dict(torch.load(
        model_path+'model_'+data_name+'_left_white.pt', map_location=device))
    model_left_pial.load_state_dict(torch.load(
        model_path+'model_'+data_name+'_left_pial.pt', map_location=device))
    model_right_white.load_state_dict(torch.load(
        model_path+'model_'+data_name+'_right_white.pt', map_location=device))
    model_right_pial.load_state_dict(torch.load(
        model_path+'model_'+data_name+'_right_pial.pt', map_location=device))
    
    
    # ------ inference ------ 
    print('Start surface reconstruction ...')
    t_start = time.time()
    with torch.no_grad():
        v_left_white = model_left_white(
            v_left_in, T, age, vol_left_in)
        v_left_pial = model_left_pial(
            v_left_white, T, age, vol_left_in)
        v_right_white = model_right_white(
            v_right_in, T, age, vol_right_in)
        v_right_pial = model_right_pial(
            v_right_white, T, age, vol_right_in)
    t_end = time.time()
    print('Finished. Runtime:{}'.format(np.round(t_end-t_start,4)))
    
    print('Save surface meshes ...', end=' ')
    # tensor to numpy
    v_left_white = v_left_white[0].cpu().numpy()
    v_left_pial = v_left_pial[0].cpu().numpy()
    f_left_in = f_left_in[0].cpu().numpy()
    v_right_white = v_right_white[0].cpu().numpy()
    v_right_pial = v_right_pial[0].cpu().numpy()
    f_right_in = f_right_in[0].cpu().numpy()
    
    # map surfaces to their original spaces
    v_left_white = v_left_white * 112 + [56, 112, 80]
    v_left_white[:,0] = v_left_white[:,0] + 64
    v_left_white = apply_affine(v_left_white, affine_t2)
    v_left_pial = v_left_pial * 112 + [56, 112, 80]
    v_left_pial[:,0] = v_left_pial[:,0] + 64
    v_left_pial = apply_affine(v_left_pial, affine_t2)
    f_left_in = f_left_in[:,[2,1,0]]
    
    v_right_white = v_right_white * 112 + [56, 112, 80]
    v_right_white = apply_affine(v_right_white, affine_t2)
    v_right_pial = v_right_pial * 112 + [56, 112, 80]
    v_right_pial = apply_affine(v_right_pial, affine_t2)
    f_right_in = f_right_in[:,[2,1,0]]

    save_gifti_surface(v_left_white, f_left_in,
                       save_path+'surf_left_white.surf.gii',
                       surf_hemi='CortexLeft', surf_type='GrayWhite')
    save_gifti_surface(v_left_pial, f_left_in,
                       save_path+'surf_left_pial.surf.gii',
                       surf_hemi='CortexLeft', surf_type='Pial')
    save_gifti_surface(v_right_white, f_right_in,
                       save_path+'surf_right_white.surf.gii',
                       surf_hemi='CortexRight', surf_type='GrayWhite')
    save_gifti_surface(v_right_pial, f_right_in,
                       save_path+'surf_right_pial.surf.gii',
                       surf_hemi='CortexRight', surf_type='Pial')
    print('Done.')
          
    