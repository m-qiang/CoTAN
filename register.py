import ants
import nibabel as nib
import numpy as np
import argparse
from scipy.io import loadmat


def affine_matrix(ants_trans):
    """convert ants transform to a 4x4 affine matrix"""
    transform = np.zeros([4,4])
    m_matrix = loadmat(
        ants_trans['fwdtransforms'][0])['AffineTransform_float_3_3'][:9].reshape(3,3)# .T
    m_center = loadmat(
        ants_trans['fwdtransforms'][0])['fixed'][:,0]
    m_translate = loadmat(
        ants_trans['fwdtransforms'][0])['AffineTransform_float_3_3'][9:][:,0]
    m_offset = m_translate + m_center - m_matrix @ m_center

    # ITK affine to affine matrix
    transform[:3,:3] = m_matrix
    transform[:3,-1] = -m_offset
    transform[3,:] = np.array([0,0,0,1])

    # LIP space to RAS
    transform[2,-1] = -transform[2,-1]
    transform[2,1] = -transform[2,1]
    transform[1,2] = -transform[1,2]
    transform[2,0] = -transform[2,0]
    transform[0,2] = -transform[0,2]
    return transform


if __name__ == "__main__":
    # load arguments
    parser = argparse.ArgumentParser(description="Affine Registration")
    parser.add_argument('--fix_path', default='./template/mni152_brain_clip.nii.gz',
                        type=str, help="directory of the fixed image")
    parser.add_argument('--move_path', default='./dataset/',
                        type=str, help="directory of the moving image")
    parser.add_argument('--save_path', default='./result/',
                        type=str, help="directory to save the aligned image")
    args = parser.parse_args()
    fix_path = args.fix_path
    move_path = args.move_path
    save_path = args.save_path
    
    # load images
    fix_img = ants.image_read(fix_path)
    affine_fix = nib.load(fix_path).affine
    move_img = ants.image_read(move_path)

    # affine registration
    ants_trans = ants.registration(
        fixed=fix_img,
        moving=move_img,
        type_of_transform='AffineFast',
        aff_metric='GC')
    
    # warp the image
    warp_img = ants.apply_transforms(
        fixed=fix_img,
        moving=move_img,
        transformlist=ants_trans['fwdtransforms'],
        interpolator='linear')

    # compute new affine matrix
    affine_mat = affine_matrix(ants_trans)
    affine_warp = affine_mat @ affine_fix

    # save file
    warp_img = nib.Nifti1Image(
        warp_img.numpy().astype(np.float32), affine_warp)
    warp_img.header['xyzt_units']=2
    nib.save(warp_img, save_path)
        