import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import argparse
import logging
from cotan import CoTAN
from util import (
    apply_affine,
    adjacency,
    edge_to_face,
    compute_vert_normal,
    compute_face_normal,
    compute_mesh_distance
)



class SurfData():
    def __init__(self, vol, v_in, v_gt,
                 f_in, f_gt, age):
        self.vol = torch.from_numpy(vol)
        self.v_in = torch.Tensor(v_in)
        self.v_gt = torch.Tensor(v_gt)
        self.f_in = torch.LongTensor(f_in)
        self.f_gt = torch.LongTensor(f_gt)
        self.age = torch.Tensor(age)

        
class SurfDataset(Dataset):
    def __init__(self, data):
        super(Dataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = self.data[i]
        return data.vol, data.v_in, data.v_gt,\
               data.f_in, data.f_gt, data.age


def load_data(args, data_split='train'):
    """load dataset for training"""
    
    # ------ load arguments ------ 
    data_path = args.data_path
    surf_hemi = args.surf_hemi
    surf_type = args.surf_type
    data_info = pd.read_csv(data_path+'combined.tsv', sep='\t')
    data_path = data_path+data_split+'/'

    subject_list = sorted(os.listdir(data_path))
    data_list = []

    if surf_type == 'white':
        # load volume template
        vol_mni = nib.load('./template/mni152_brain_clip.nii.gz')
        
    for i in tqdm(range(len(subject_list))):
        subid = subject_list[i]
        
        # ------- load participants information and age ------- 
        parid, sesid = subid.split('_')
        parid = parid[4:]
        sesid = sesid[4:]
        age = data_info.loc[(data_info['participant_id'] == parid) &\
                            (data_info['session_id'] == int(sesid))]['scan_age'].to_numpy()
        age = (age-20) / 30.  # normalize age
        
        # ------- load input volume  ------- 
        vol = nib.load(data_path+subid+'/'+subid+'_T2w_brain_affine.nii.gz')
        vol_arr = vol.get_fdata()
        vol_arr = (vol_arr / 40.).astype(np.float32)  # normalize intensity

        # ------- load surface -------
        if surf_type == 'white':
            surf_in = nib.load('./template/init_surf_'+surf_hemi+'.surf.gii')
            surf_gt = nib.load(data_path+subid+'/'+subid+'_'+surf_hemi+'_white.surf.gii')
            affine_in = vol_mni.affine
            affine_gt = vol.affine
        elif surf_type == 'pial':
            surf_in = nib.load(data_path+subid+'/'+subid+'_'+surf_hemi+'_white.surf.gii')
            surf_gt = nib.load(data_path+subid+'/'+subid+'_'+surf_hemi+'_pial.surf.gii')
            affine_in = vol.affine
            affine_gt = vol.affine
            
        # ------- process surface -------
        v_in, f_in = surf_in.agg_data('pointset'), surf_in.agg_data('triangle')
        v_gt, f_gt = surf_gt.agg_data('pointset'), surf_gt.agg_data('triangle')
        v_in = apply_affine(v_in, np.linalg.inv(affine_in))
        v_gt = apply_affine(v_gt, np.linalg.inv(affine_gt))
        f_in = f_in[:,[2,1,0]]
        f_gt = f_gt[:,[2,1,0]]
            
        # ------- clip data -------
        if surf_hemi == 'left':
            vol_arr = vol_arr[64:]
            v_in[:,0] = v_in[:,0] - 64
            v_gt[:,0] = v_gt[:,0] - 64
        elif surf_hemi == 'right':
            vol_arr = vol_arr[:112]
        v_in = (v_in - [56, 112, 80]) / 112
        v_gt = (v_gt - [56, 112, 80]) / 112
        surf_data = SurfData(vol=vol_arr[None].astype(np.float32),
                             v_in=v_in.astype(np.float32),
                             v_gt=v_gt.astype(np.float32),
                             f_in=f_in, f_gt=f_gt, age=age)
        data_list.append(surf_data)  # add to data list
        
    # make dataset
    surf_dataset = SurfDataset(data_list)
    return surf_dataset


def train_loop(args):
    # ------ load arguments ------ 
    model_path = args.model_path  # path to save the model checkpoints
    surf_type = args.surf_type  # white or pial
    surf_hemi = args.surf_hemi  # left or right
    data_name = args.data_name  # dhcp
    tag = args.tag  # identity of experiments
    device = torch.device(args.device)
    n_epoch = args.n_epoch  # training epochs
    lr = args.lr  # learning rate
    step_size = args.step_size
    if surf_type == 'white':
        layers = [16,32,64,128,128]  # number of channels for each layer
    elif surf_type == 'pial':
        layers = [16,32,32,32,32]  # fewer params to avoid overfitting
    M = args.n_svf  # number of velocity fields
    R = args.n_res  # number of resolutions
    n_sample = args.n_sample  # number of sampled points for training
    weight_nc = args.weight_nc  # weight for nc loss
    weight_lap = args.weight_lap  # weight for lap loss
    weight_decay = args.weight_decay
    print(lr, step_size, M, R)
    print(weight_nc, weight_lap, weight_decay)
    
    # start training logging
    logging.basicConfig(filename=model_path+'log_'+data_name+'_'+surf_hemi\
                                 +'_'+surf_type+'_'+tag+'.log',
                        level=logging.INFO, format='%(asctime)s %(message)s')
    
    # ------ load dataset ------ 
    logging.info("load dataset ...")
    trainset = load_data(args, data_split='train')
    validset = load_data(args, data_split='valid')

    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)
    
    # ------ pre-compute for loss ------ 
    if surf_type == 'white':
        _, _, _, f_in, _, _ = next(iter(trainloader))
        f_in = f_in.to(device)
        e2f = edge_to_face(f_in)  # for normal consistency loss
        adj_matrix, adj_degree = adjacency(f_in)  # for laplacian loss
    # input integration time sequence
    T = torch.arange(1./step_size).to(device).unsqueeze(1) * step_size
    
    # ------ initialize model ------ 
    logging.info("initalize model ...")
    model = CoTAN(layers=layers, M=M, R=R).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ------ training loop ------ 
    logging.info("start training ...")
    for epoch in tqdm(range(n_epoch+1)):
        avg_loss = []
        
        if surf_type == 'white' and epoch == int(n_epoch//2+1):
            # start fine-tuning
            logging.info("epoch:{}, start fine-tuning ...".format(epoch))
            optimizer = optim.Adam(model.parameters(), lr=weight_decay*lr)
            weight_nc = weight_decay * weight_nc
            weight_lap = weight_decay * weight_lap
            print(weight_nc, weight_lap)
            
        for idx, data in enumerate(trainloader):
            vol_in, v_in, v_gt, f_in, f_gt, age = data
            vol_in = vol_in.to(device).float()
            v_in = v_in.to(device)
            f_in = f_in.to(device)
            v_gt = v_gt.to(device)
            f_gt = f_gt.to(device)
            age = age.repeat(int(1./step_size), 1).to(device)
            
            optimizer.zero_grad()
            v_pred = model(v_in, T, age, vol_in)
            
            if surf_type == 'white':
                # normal consistency loss
                n_f = compute_face_normal(v_pred, f_in)  # face normal
                nc_loss = (1 - n_f[:,e2f].prod(-2).sum(-1)).mean()
                # laplacian loss
                v_lap = v_pred - adj_matrix.bmm(v_pred) / adj_degree
                lap_loss = (v_lap**2).sum(-1).mean()
                # reconstruction loss
                if epoch >= int(n_epoch//2+1):  # sample points for fine-tuning
                    mesh_pred = Meshes(verts=v_pred, faces=f_in)
                    mesh_gt = Meshes(verts=v_gt, faces=f_gt)
                    v_pred = sample_points_from_meshes(mesh_pred, num_samples=n_sample)
                    v_gt = sample_points_from_meshes(mesh_gt, num_samples=n_sample)
                recon_loss = chamfer_distance(v_pred, v_gt)[0]  # vertices for pre-train
                # total loss
                loss = 1e3 * (recon_loss + weight_nc*nc_loss + weight_lap*lap_loss)
            elif surf_type == 'pial':
                # mse loss
                loss = 1e3 * nn.MSELoss()(v_pred, v_gt)
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        logging.info("epoch:{}, loss:{}".format(epoch, np.mean(avg_loss)))

        if epoch % 10 == 0:  # start validation
            logging.info('------------ validation ------------')
            with torch.no_grad():
                valid_chamfer = []
                valid_assd = []
                valid_hd = []
                for idx, data in enumerate(validloader):
                    vol_in, v_in, v_gt, f_in, f_gt, age = data
                    vol_in = vol_in.to(device).float()
                    v_in = v_in.to(device)
                    f_in = f_in.to(device)
                    v_gt = v_gt.to(device)
                    f_gt = f_gt.to(device)
                    age = age.repeat(int(1./step_size), 1).to(device)
                    v_pred = model(v_in, T, age, vol_in)
                
                    # compute chamfer distance
                    chamfer = chamfer_distance(v_pred, v_gt)[0].item()
                    # compute ASSD and HD
                    # note that this is NOT the final ASSD and HD for the inference,
                    # because the surfaces have not been mapped to its original space
                    assd, hd = compute_mesh_distance(v_pred, v_gt, f_in, f_gt)
                    valid_chamfer.append(1e3 * chamfer)
                    valid_assd.append(1e2 * assd)
                    valid_hd.append(1e2 * hd)
                    
                logging.info('epoch:{}'.format(epoch))
                logging.info('chamfer:{}'.format(np.mean(valid_chamfer)))
                logging.info('assd:{}'.format(np.mean(valid_assd)))
                logging.info('hd:{}'.format(np.mean(valid_hd)))
                logging.info('-------------------------------------')
                
                # save model checkpoints
                torch.save(model.state_dict(), model_path+'model_'+data_name+'_'+surf_hemi\
                           +'_'+surf_type+'_'+tag+'_'+str(epoch)+'epochs.pt')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="CoTAN")
    
    parser.add_argument('--data_path', default='./dataset/', type=str, help="directory of the dataset")
    parser.add_argument('--model_path', default='./model/', type=str, help="directory to save the model")
    parser.add_argument('--data_name', default='dhcp', type=str, help="[dhcp, ...]")
    parser.add_argument('--surf_type', default='white', type=str, help="[white, pial]")
    parser.add_argument('--surf_hemi', default='left', type=str, help="[left, right]")
    parser.add_argument('--device', default="cuda", type=str, help="cuda or cpu")
    parser.add_argument('--tag', default='0000', type=str, help="identity of experiments")
    parser.add_argument('--step_size', default=0.02, type=float, help="integration step size")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--n_epoch', default=200, type=int, help="number of training epochs")
    parser.add_argument('--n_svf', default=4, type=int, help="number of velocity fields")
    parser.add_argument('--n_res', default=3, type=int, help="number of scales")
    parser.add_argument('--n_sample', default=150000, type=int, help="number of sampled points")
    parser.add_argument('--weight_nc', default=5e-4, type=float, help="weight of normal consistency loss")
    parser.add_argument('--weight_lap', default=0.5, type=float, help="weight of Laplacian loss")
    parser.add_argument('--weight_decay', default=0.2, type=float, help="weight decay for regularization")

    args = parser.parse_args()
    
    train_loop(args)