import os
import glob
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import scipy.ndimage
from datetime import datetime
import matplotlib.pyplot as plt
import pdb

# data_path = '/data/jiahong/data/FDG_PET_selected/'
# data_path2 = '/data/jiahong/data/FDG_PET_selected_old/'
# subj_paths = glob.glob(data_path+'*') + glob.glob(data_path2+'*')

data_path1 = '/data/jiahong/data/BraTS/MICCAI_BraTS2020_TrainingData/'
data_path2 = '/data/jiahong/data/BraTS/MICCAI_BraTS2020_ValidationData/'
subj_paths = glob.glob(data_path1+'*') + glob.glob(data_path2+'*')

pdb.set_trace()


# PET_list = []
T1_list = []
T1c_list = []
T2_list = []
T2_FLAIR_list = []
seg_list = []
subj_id_list = []
subj_all_dict = {}
for subj_path in subj_paths:
    if os.path.isfile(subj_path):
        continue
    subj_id = os.path.basename(subj_path)
    print(subj_id)
    subj_id_num = subj_id.split('_')[2]
    if len(os.listdir(subj_path)) == 0:
        continue
    subj_id_list.append(subj_id)
    subj_dict = {}
    if 'Training' in subj_path:
        predix = 'BraTS20_Training_'
    else:
        predix = 'BraTS20_Validation_'
    if os.path.exists(os.path.join(subj_path, predix+subj_id_num+'_seg.nii.gz')):
        subj_dict['seg'] = os.path.join(subj_path, predix+subj_id_num+'_seg.nii.gz')
        seg_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, predix+subj_id_num+'_t1.nii.gz')):
        subj_dict['T1'] = os.path.join(subj_path, predix+subj_id_num+'_t1.nii.gz')
        T1_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, predix+subj_id_num+'_t1ce.nii.gz')):
        subj_dict['T1c'] = os.path.join(subj_path, predix+subj_id_num+'_t1ce.nii.gz')
        T1c_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, predix+subj_id_num+'_t2.nii.gz')):
        subj_dict['T2'] = os.path.join(subj_path, predix+subj_id_num+'_t2.nii.gz')
        T2_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, predix+subj_id_num+'_flair.nii.gz')):
        subj_dict['T2_FLAIR'] = os.path.join(subj_path, predix+subj_id_num+'_flair.nii.gz')
        T2_FLAIR_list.append(subj_path)
    subj_all_dict[subj_id] = subj_dict

print('Total:', len(subj_id_list))
print('T1:', len(T1_list))
print('T1c:', len(T1c_list))
print('T2:', len(T2_list))
print('T2_FLAIR:', len(T2_FLAIR_list))
print('Seg:', len(seg_list))
pdb.set_trace()

# h5_path = '/data/jiahong/project_zerodose_pytorch/data/BraTS_All_zscore_10.h5'
# f  =h5py.File(h5_path, 'a')
# subj_data_dict = {}
# subj_id_list = []
# for i, subj_id in enumerate(subj_all_dict.keys()):
#     subj_data = f.create_group(subj_id)
#     subj_dict = subj_all_dict[subj_id]
#     for contrast_name in subj_dict.keys():
#         img_nib = nib.load(subj_dict[contrast_name])
#         img = img_nib.get_fdata()
#         if img.shape != (240, 240, 155) or np.nanmax(img) == 0 or np.isnan(img[:,:,50:-50]).sum()>100000:
#             print(subj_id)
#             print(img.shape, np.nanmax(img), np.isnan(img[:,:,50:-50]).sum())
#             break
#         img = np.nan_to_num(img, nan=0.)
#         img = img[40:-40, 24:-24]           # (240,240,155)->(160,192,155)
#         if contrast_name != 'seg':
#             brain_mask = (img>0).astype(int)
#             # pdb.set_trace()
#             img_pos_num = (img > 0).sum()
#             norm = img.sum() / (img_pos_num+1)
#             # img = img / norm  # norm by dividing mean
#             std = np.sqrt((brain_mask * (img - norm)**2).sum() / (img_pos_num+1))
#             img = (img - norm) / (std + 1e-8)
#             # print(img.min())
#             img[brain_mask == 0] = -10
#         subj_data.create_dataset(contrast_name, data=img)
#     subj_id_list.append(subj_id)
#     print(i, subj_id)

np.random.seed(10)
num_subj = len(subj_id_list)
np.random.shuffle(subj_id_list)

def save_data_txt(path, subj_id_list):
    count = 0
    with open(path, 'w') as ft:
        for subj_id in subj_id_list:
            for i in range(50, 155-50):
                ft.write(subj_id+' '+str(i)+'\n')
                count += 1
    print(count)

def save_data_txt_3d(path, subj_id_list):
    count = 0
    with open(path, 'w') as ft:
        for subj_id in subj_id_list:
            ft.write(subj_id+'\n')
            count += 1
    print(count)

def remove_validation_data(subj_id_list):
    subj_id_list_sel = []
    for subj_id in subj_id_list:
        if 'Validation' not in subj_id:
            subj_id_list_sel.append(subj_id)
    return subj_id_list_sel

# pdb.set_trace()
for fold in range(5):
    # pdb.set_trace()
    subj_id_list_test = subj_id_list[fold*int(0.2*num_subj):(fold+1)*int(0.2*num_subj)]
    subj_id_list_train_val = subj_id_list[:fold*int(0.2*num_subj)] + subj_id_list[(fold+1)*int(0.2*num_subj):]
    subj_id_list_val = subj_id_list_train_val[:int(0.1*len(subj_id_list_train_val))]
    subj_id_list_train = subj_id_list_train_val[int(0.1*len(subj_id_list_train_val)):]
    pdb.set_trace()

    subj_id_list_train = remove_validation_data(subj_id_list_train)
    subj_id_list_val = remove_validation_data(subj_id_list_val)
    subj_id_list_test = remove_validation_data(subj_id_list_test)

    # save_data_txt('../data/fold_BraTS_'+str(fold)+'_train_noval.txt', subj_id_list_train)
    # save_data_txt('../data/fold_BraTS_'+str(fold)+'_val_noval.txt', subj_id_list_val)
    # save_data_txt('../data/fold_BraTS_'+str(fold)+'_test_noval.txt', subj_id_list_test)
    save_data_txt_3d('../data/fold_BraTS_3d_'+str(fold)+'_train_noval.txt', subj_id_list_train)
    save_data_txt_3d('../data/fold_BraTS_3d_'+str(fold)+'_val_noval.txt', subj_id_list_val)
    save_data_txt_3d('../data/fold_BraTS_3d_'+str(fold)+'_test_noval.txt', subj_id_list_test)
