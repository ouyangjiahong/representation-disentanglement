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

data_path = '/data/jiahong/data/FDG_PET_selected_checked/'
subj_paths = glob.glob(data_path+'*')

tumor_subj_list = ['case_0103', 'case_0110', 'case_0111', 'case_0117', 'case_0118', 'case_0124',
                    'case_0129', 'case_0132', 'case_0136', 'case_0139', 'case_0142', 'case_0144',
                    'case_0145', 'case_0146', 'case_0148', 'case_0149', 'case_0151', 'case_0156',
                    'case_0157', 'case_0158', 'case_0160', 'case_0161', 'case_0164', 'case_0168',
                    'case_0169', 'case_0174', 'case_0175', 'case_0188', 'case_0196', 'case_0208',
                    'case_0219', 'case_0226', 'case_0238', 'case_0240', 'case_0241', 'case_0242',
                    'case_0248', 'case_0252', 'case_0253', 'case_0257', 'case_0258', 'Fdg_Stanford_001',
                    'Fdg_Stanford_002', 'Fdg_Stanford_003', 'Fdg_Stanford_004', 'Fdg_Stanford_005',
                    'Fdg_Stanford_006', 'Fdg_Stanford_010', 'Fdg_Stanford_011', 'Fdg_Stanford_012',
                    'Fdg_Stanford_014', 'Fdg_Stanford_015', 'Fdg_Stanford_016', 'Fdg_Stanford_017',
                    'Fdg_Stanford_023', 'Fdg_Stanford_024', 'Fdg_Stanford_025', 'Fdg_Stanford_026',
                    'Fdg_Stanford_027', 'Fdg_Stanford_028', 'Fdg_Stanford_029', 'Fdg_Stanford_030',
                    'Fdg_Stanford_031', 'Fdg_Stanford_032', '852_06182015', '869_06252015', '1496_05272016',
                    '1498_05312016', '1512_06062016', '1549_06232016', '1559_06282016', '1572_07052016',
                    '1604_07142016', '1610_07152016', '1619_07192016', '1657_08022016', '2002_02142017',
                    '2010_02212017', '2114_04102017', '2120_04122017', '2142_04242017', '2275_07062017',
                    '2277_07072017', '2284_07112017', '2292_07122017', '2295_07132017', '2310_07242017',
                    '2374_08242017', '2377_08252017', '2388_08302017']
demantia_subj_list = ['case_0106', 'case_0123', 'case_0130', 'case_0135', 'case_0137', 'case_0141',
                    'case_0170', 'case_0184', 'case_0193', 'case_0195', 'case_0198', 'case_0209',
                    'case_0215', 'case_0216', 'case_0224', 'Fdg_Stanford_013']

# tumor_subj_full_list = []
# nifti_names = [os.path.basename(name) for name in glob.glob('/data/jiahong/project_zerodose_pytorch/src/nifti/*')]
# for subj_id in tumor_subj_list:
#     if subj_id+'_PET.nii' in nifti_names:
#         tumor_subj_full_list.append(subj_id)
# print(tumor_subj_full_list)
# pdb.set_trace()

PET_list = []
T1_list = []
T1c_list = []
T2_FLAIR_list = []
ASL_list = []
subj_id_list = []
subj_all_dict = {}
for subj_path in subj_paths:
    subj_id = os.path.basename(subj_path)
    if len(os.listdir(subj_path)) == 0:
        continue
    subj_id_list.append(subj_id)
    subj_dict = {}
    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_PET.nii')):
        subj_dict['PET'] = os.path.join(subj_path, 'tpm_r2T1_PET.nii')
        PET_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, 'tpm_T1.nii')):
        subj_dict['T1'] = os.path.join(subj_path, 'tpm_T1.nii')
        T1_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_T1c.nii')):
        subj_dict['T1c'] = os.path.join(subj_path, 'tpm_r2T1_T1c.nii')
        T1c_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_T2_FLAIR.nii')):
        subj_dict['T2_FLAIR'] = os.path.join(subj_path, 'tpm_r2T1_T2_FLAIR.nii')
        T2_FLAIR_list.append(subj_path)
    if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_r2PET_ASL.nii')):
        subj_dict['ASL'] = os.path.join(subj_path, 'tpm_r2T1_r2PET_ASL.nii')
        ASL_list.append(subj_path)
    subj_all_dict[subj_id] = subj_dict
    # if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_PET.nii')):
    #     subj_id_list.append(subj_id)
    #     subj_dict = {'PET': os.path.join(subj_path, 'tpm_r2T1_PET.nii')}
    #     if os.path.exists(os.path.join(subj_path, 'tpm_T1.nii')):
    #         subj_dict['T1'] = os.path.join(subj_path, 'tpm_T1.nii')
    #         T1_list.append(subj_path)
    #     if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_T1c.nii')):
    #         subj_dict['T1c'] = os.path.join(subj_path, 'tpm_r2T1_T1c.nii')
    #         T1c_list.append(subj_path)
    #     if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_T2_FLAIR.nii')):
    #         subj_dict['T2_FLAIR'] = os.path.join(subj_path, 'tpm_r2T1_T2_FLAIR.nii')
    #         T2_FLAIR_list.append(subj_path)
    #     if os.path.exists(os.path.join(subj_path, 'tpm_r2T1_r2PET_ASL.nii')):
    #         subj_dict['ASL'] = os.path.join(subj_path, 'tpm_r2T1_r2PET_ASL.nii')
    #         ASL_list.append(subj_path)
    #     subj_all_dict[subj_id] = subj_dict

print('Total:', len(subj_id_list))
print('T1:', len(T1_list))
print('T1c:', len(T1c_list))
print('T2_FLAIR:', len(T2_FLAIR_list))
print('ASL:', len(ASL_list))
print('PET:', len(PET_list))
pdb.set_trace()

brain_mask_nib = nib.load(os.path.join('/data/jiahong/project_zerodose_pytorch/data/', 'tpm_mask_new.nii'))
brain_mask = brain_mask_nib.get_fdata()

# h5_path = '/data/jiahong/project_zerodose_pytorch/data/ZeroDose_FDG_All_1103_zscore_10.h5'
h5_path = '/data/jiahong/project_zerodose_pytorch/data/ZeroDose_FDG_All_1103_zscore_10_norm.h5'
# h5_path = '/data/jiahong/project_zerodose_pytorch/data/ZeroDose_FDG_All_1103_norm.h5'
f  = h5py.File(h5_path, 'a')
subj_data_dict = {}
subj_id_list = []
for i, subj_id in enumerate(subj_all_dict.keys()):
    subj_data = f.create_group(subj_id)
    subj_dict = subj_all_dict[subj_id]
    for contrast_name in subj_dict.keys():
        # if contrast_name != 'PET':
        #     continue
        img_nib = nib.load(subj_dict[contrast_name])
        img = img_nib.get_fdata()
        if img.shape != (157, 189, 156) or np.nanmax(img) == 0 or np.isnan(img[:,:,20:-20]).sum()>100000:
            print(subj_id)
            print(img.shape, np.nanmax(img), np.isnan(img[:,:,20:-20]).sum())
            break
        img = np.nan_to_num(img, nan=0.)
        img = img * brain_mask
        # img = img / (img.mean()+1e-6)
        img[img<0] = 0

        img_pos_num = (img > 0).sum()                   # new, 11.3
        norm = img.sum() / (img_pos_num+1)
        # img = img / norm  # norm by dividing mean
        std = np.sqrt((brain_mask * (img - norm)**2).sum() / (img_pos_num+1))
        img = (img - norm) / (std + 1e-8)
        img[brain_mask == 0] = -10
        # pdb.set_trace()
        img = np.concatenate([img, -10*np.ones((3,189,156))], 0)     # (157,189) -> (160,192)
        img = np.concatenate([img, -10*np.ones((160,3,156))], 1)

        # subj_data.create_dataset(contrast_name, data=img)
        # subj_data.create_dataset(contrast_name, data=norm)
        # pdb.set_trace()
        subj_data.create_dataset(contrast_name, data=[norm, std])
    subj_id_list.append(subj_id)
    print(i, subj_id)

np.random.seed(10)
num_subj = len(subj_id_list)
np.random.shuffle(subj_id_list)

def save_data_txt(path, subj_id_list):
    count = 0
    with open(path, 'w') as ft:
        for subj_id in subj_id_list:
            for i in range(20, 156-20):
                ft.write(subj_id+' '+str(i)+'\n')
                count += 1
    print(count)

def save_data_txt_allslices(path, subj_id_list):
    count = 0
    with open(path, 'w') as ft:
        for subj_id in subj_id_list:
            for i in range(0, 156):
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

def select_complete_subj(f, subj_id_list):
    subj_id_list_sel = []
    for subj_id in subj_id_list:
        # if 'T1' in f[subj_id].keys() and 'T2_FLAIR' in f[subj_id].keys() and 'PET' in f[subj_id].keys():
        if 'T1' in f[subj_id].keys() and 'T1c' in f[subj_id].keys() and 'T2_FLAIR' in f[subj_id].keys() and 'ASL' in f[subj_id].keys() and 'PET' in f[subj_id].keys():
        # if 'T1' in f[subj_id].keys() and 'T2_FLAIR' in f[subj_id].keys() and 'ASL' in f[subj_id].keys() and 'PET' in f[subj_id].keys():
            # if subj_id in tumor_subj_list:
            #     subj_id_list_sel.append(subj_id)
            subj_id_list_sel.append(subj_id)
    return subj_id_list_sel


h5_path = '/data/jiahong/project_zerodose_pytorch/data/ZeroDose_FDG_All_1103.h5'
f  = h5py.File(h5_path, 'r')
#

subj_id_list_sel = select_complete_subj(f, subj_id_list)
print(len(subj_id_list), len(subj_id_list_sel))
pdb.set_trace()
num_subj = len(subj_id_list_sel)

for fold in range(5):
    subj_id_list_test = subj_id_list_sel[fold*int(0.2*num_subj):(fold+1)*int(0.2*num_subj)]
    subj_id_list_train_val = subj_id_list_sel[:fold*int(0.2*num_subj)] + subj_id_list_sel[(fold+1)*int(0.2*num_subj):]
    subj_id_list_val = subj_id_list_train_val[:int(0.1*len(subj_id_list_train_val))]
    subj_id_list_train = subj_id_list_train_val[int(0.1*len(subj_id_list_train_val)):]

    subj_id_list_test = select_complete_subj(f, subj_id_list_test)
    subj_id_list_val = select_complete_subj(f, subj_id_list_val)
    subj_id_list_train = select_complete_subj(f, subj_id_list_train)

#     # save_data_txt('../data/fold'+str(fold)+'_train_1103.txt', subj_id_list_train)
#     # save_data_txt('../data/fold'+str(fold)+'_val_1103.txt', subj_id_list_val)
#     # save_data_txt('../data/fold'+str(fold)+'_test_1103.txt', subj_id_list_test)
#     save_data_txt('../data/fold'+str(fold)+'_train_1103_sel.txt', subj_id_list_train)
#     save_data_txt('../data/fold'+str(fold)+'_val_1103_sel.txt', subj_id_list_val)
#     save_data_txt('../data/fold'+str(fold)+'_test_1103_sel.txt', subj_id_list_test)

    # save_data_txt('../data/fold'+str(fold)+'_train_3contrasts_sel.txt', subj_id_list_train)
    # save_data_txt('../data/fold'+str(fold)+'_val_3contrasts_sel.txt', subj_id_list_val)
    # save_data_txt('../data/fold'+str(fold)+'_test_3contrasts_sel.txt', subj_id_list_test)
    save_data_txt('../data/fold'+str(fold)+'_train_4contrasts_sel_all.txt', subj_id_list_train)
    save_data_txt('../data/fold'+str(fold)+'_val_4contrasts_sel_all.txt', subj_id_list_val)
    save_data_txt('../data/fold'+str(fold)+'_test_4contrasts_sel_all.txt', subj_id_list_test)

    # save_data_txt_allslices('../data/fold'+str(fold)+'_test_4contrasts_sel_tumor_allslices.txt', subj_id_list_test)

# save_data_txt_3d('../data/ZeroDose_3d_all.txt', subj_id_list_sel)
