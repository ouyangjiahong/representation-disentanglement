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


data_path = '/data/jiahong/data/NCANDA/'
T1_path = data_path + 'T1/'
T2_path = data_path + 'T2/'
subj_paths_T1 = glob.glob(T1_path+'*')
subj_paths_T1 = [os.path.basename(subj_path) for subj_path in subj_paths_T1]
subj_paths_T2 = glob.glob(T2_path+'*')
subj_paths_T2 = [os.path.basename(subj_path) for subj_path in subj_paths_T2]

subj_paths = list(set(subj_paths_T1) & set(subj_paths_T2))
print('Total:', len(subj_paths))
pdb.set_trace()

subj_id_list = []
subj_all_dict = {}
for subj_path in subj_paths:
    subj_id = os.path.basename(subj_path)
    subj_id_list.append(subj_id)
    subj_dict = {}
    subj_dict['T1'] = os.path.join(T1_path, subj_id)
    subj_dict['T2'] = os.path.join(T2_path, subj_id)
    subj_all_dict[subj_id] = subj_dict

# pdb.set_trace()

# h5_path = '/data/jiahong/project_zerodose_pytorch/data/NCANDA_All_zscore_10.h5'
# f  =h5py.File(h5_path, 'a')
# subj_data_dict = {}
# for i, subj_id in enumerate(subj_all_dict.keys()):
#     if subj_id not in f.keys():
#         subj_data = f.create_group(subj_id)
#         subj_dict = subj_all_dict[subj_id]
#         for contrast_name in subj_dict.keys():
#             img_nib = nib.load(subj_dict[contrast_name])
#             img = img_nib.get_fdata()
#             if img.shape != (240, 240, 240) or np.nanmax(img) == 0 or np.isnan(img[:,:,50:-50]).sum()>100000:
#                 print(subj_id)
#                 print(img.shape, np.nanmax(img), np.isnan(img[:,:,50:-50]).sum())
#                 break
#             img = np.nan_to_num(img, nan=0.)
#             img = img[40:-40, 24:-24, 40:-40]           # (240,240,240)->(160,192,160)
#
#             brain_mask = (img>0).astype(int)
#             # pdb.set_trace()
#             img_pos_num = (img > 0).sum()
#             norm = img.sum() / (img_pos_num+1)
#             # img = img / norm  # norm by dividing mean
#             std = np.sqrt((brain_mask * (img - norm)**2).sum() / (img_pos_num+1))
#             img = (img - norm) / (std + 1e-8)
#             # print(img.min())
#             img[brain_mask == 0] = -10
#
#             subj_data.create_dataset(contrast_name, data=img)
#         print(i, subj_id)

np.random.seed(10)
num_subj = len(subj_id_list)
np.random.shuffle(subj_id_list)

def save_data_txt(path, subj_id_list):
    count = 0
    with open(path, 'w') as ft:
        for subj_id in subj_id_list:
            for i in range(60, 160-60):
                ft.write(subj_id+' '+str(i)+'\n')
                count += 1
    print(count)


for fold in range(5):
    subj_id_list_test = subj_id_list[fold*int(0.2*num_subj):(fold+1)*int(0.2*num_subj)]
    subj_id_list_train_val = subj_id_list[:fold*int(0.2*num_subj)] + subj_id_list[(fold+1)*int(0.2*num_subj):]
    subj_id_list_val = subj_id_list_train_val[:int(0.1*len(subj_id_list_train_val))]
    subj_id_list_train = subj_id_list_train_val[int(0.1*len(subj_id_list_train_val)):]

    pdb.set_trace()

    save_data_txt('../data/fold_NCANDA_'+str(fold)+'_train.txt', subj_id_list_train)
    save_data_txt('../data/fold_NCANDA_'+str(fold)+'_val.txt', subj_id_list_val)
    save_data_txt('../data/fold_NCANDA_'+str(fold)+'_test.txt', subj_id_list_test)
