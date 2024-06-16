# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import math
import matplotlib.pyplot as plt
import numpy as np
import openpyxl

import os
from os.path import join
import shutil
import nibabel as nib
import numpy as np


from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy,color_fa



bundles = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6',
                   'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'MLF_left', 'MLF_right', 'FPT_left',
                   'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left',
                   'ILF_right', 'MCP', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right',
                   'SLF_I_left', 'SLF_I_right', 'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right',
                   'STR_left', 'STR_right', 'UF_left', 'UF_right', 'CC', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left',
                   'T_PREM_right', 'T_PREC_left', 'T_PREC_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PAR_left',
                   'T_PAR_right', 'T_OCC_left', 'T_OCC_right', 'ST_FO_left', 'ST_FO_right', 'ST_PREF_left',
                   'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_POSTC_left',
                   'ST_POSTC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_OCC_left', 'ST_OCC_right']
#100 HCP
HCP_subjects =["992774", "991267", "987983", "984472", "983773", "979984", "978578", "965771", "965367", "959574",
               "958976", "957974", "951457", "932554", "930449", "922854", "917255", "912447", "910241", "904044",
               "901442", "901139", "901038", "899885", "898176", "896879", "896778", "894673", "889579", "877269",
               "877168", "872764", "872158", "871964", "871762", "865363", "861456", "859671", "857263", "856766",
               "849971", "845458", "837964", "837560", "833249", "833148", "826454", "826353", "816653", "814649",
               "802844", "792766", "792564", "789373", "786569", "784565", "782561", "779370", "771354", "770352",
               "765056", "761957", "759869", "756055", "753251", "751348", "749361", "748662", "748258", "742549",
               "734045", "732243", "729557", "729254", "715647", "715041", "709551", "705341", "704238", "702133",
               "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968", "673455", "672756",
               "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236", "620434", "613538"]
#
# subjects = ["992774", "991267", "987983", "984472", "983773", "979984", "978578", "965771", "965367", "959574",
#         "958976", "957974", "951457", "932554", "930449", "922854", "917255", "912447", "910241",
#         "904044", "901442", "901139", "901038", "899885", "898176", "896879", "896778", "894673", "889579",
#         "877269", "877168", "872764", "872158", "871964", "871762", "865363", "861456", "859671",
#         "857263", "856766", "849971", "845458", "837964", "837560", "833249", "833148", "826454", "826353",
#         "816653", "814649", "802844", "792766", "792564", "789373", "786569", "784565", "782561", "779370",
#         "771354", "770352", "765056", "761957", "759869", "756055", "753251", "751348", "749361", "748662",
#         "748258", "742549", "734045", "732243", "729557", "729254", "715647", "715041", "709551", "705341",
#         "704238", "702133", "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968",
#         "673455", "672756", "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236",
#         "620434", "613538", "601127", "599671", "599469"]  #HCP103


# subjects = ["992774", "991267", "987983", "984472", "983773",  "978578", "965771", "965367", "959574",
#         "958976", "957974", "932554", "930449", "922854", "917255", "912447", "910241",
#         "904044", "901442", "901139",  "899885", "898176", "896879", "896778", "894673", "889579",
#         "877269", "877168", "872158", "871964", "871762", "865363", "861456", "859671",
#         "857263", "856766", "849971", "845458", "837964", "837560", "833249", "833148", "826454", "826353",
#         "816653", "802844", "792564", "789373", "786569", "784565", "782561", "779370",
#         "771354", "770352", "765056", "761957", "759869", "756055", "751348", "748662",
#         "748258", "742549", "734045", "732243", "729557", "729254", "715647", "709551", "705341",
#         "704238", "702133", "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968",
#         "673455", "672756", "665254", "654754", "645551", "638049", "627549", "623844", "622236",
#         "613538", "599671", "599469"]  #HCP91----gradient288






## for hcp_12 novel tracts


## extract 12 tracts
bundles_novel_12 = ['CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'ILF_left', 'ILF_right','OR_left', 'OR_right', 'POPT_left', 'POPT_right',  'UF_left', 'UF_right']
bundles_index_12 = [14, 15, 18, 19, 26, 27, 29, 30, 31, 32, 43, 44]

## extract 6 tracts
bundles_novel_6 = ['CST_left', 'CST_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right']
bundles_index_6 = [14, 15, 29, 30, 31, 32]

## extract 4 tracts
bundles_novel_4 = ['CST_left', 'CST_right','OR_left', 'OR_right']
bundles_index_4 = [14, 15, 29, 30]


def extract_bundle_hcp():
    dir_list = ['/data4/wanliu/HCP_1.25mm_270/HCP_for_training_COPY', '/data4/wanliu/HCP_1.25mm_270/HCP_preproc']
    # dir = '/data4/wanliu/HCP_2.5mm_341k/HCP_preproc'
    for dir in dir_list:
        for sub in HCP_subjects:
            print(sub)
            all_bund_file = os.path.join(dir, sub, 'bundle_masks_72.nii.gz')
            bund_nii =nib.load(all_bund_file)
            affine=bund_nii.affine
            header=bund_nii.header
            bund_data = bund_nii.get_fdata()

            tract60_index = np.array(list(set(list(range(72)))-set(bundles_index_12))).astype(int)
            print(tract60_index)
            bund_60 = bund_data[:,:,:,tract60_index].astype(np.float32)
            bundle_nii_60 = nib.Nifti1Image(bund_60, affine, header)
            nib.save(bundle_nii_60, os.path.join(dir, sub, 'bundle_masks_60.nii.gz'))

            novel_index_12 = np.array(bundles_index_12).astype(int)
            bund_12 = bund_data[:,:,:,novel_index_12].astype(np.float32)
            bundle_nii_12 = nib.Nifti1Image(bund_12, affine, header)
            nib.save(bundle_nii_12, os.path.join(dir, sub, 'bundle_masks_12.nii.gz'))

            novel_index_6 = np.array(bundles_index_6).astype(int)
            bund_6 = bund_data[:,:,:,novel_index_6].astype(np.float32)
            bundle_nii_6 = nib.Nifti1Image(bund_6, affine, header)
            nib.save(bundle_nii_6, os.path.join(dir, sub, 'bundle_masks_6.nii.gz'))

            novel_index_4 = np.array(bundles_index_4).astype(int)
            bund_4 = bund_data[:,:,:,novel_index_4].astype(np.float32)
            bundle_nii_4 = nib.Nifti1Image(bund_4, affine, header)
            nib.save(bundle_nii_4, os.path.join(dir, sub, 'bundle_masks_4.nii.gz'))


def extract_bundle_hcp1():
    dir_list = ['/data4/wanliu/HCP_2.5mm_341k/HCP_for_training_COPY', '/data4/wanliu/HCP_2.5mm_341k/HCP_preproc']
    for dir in dir_list:
        for sub in HCP_subjects:
            print(sub)
            all_bund_file = os.path.join(dir, sub, 'bundle_masks_72.nii.gz')
            bund_nii =nib.load(all_bund_file)
            affine=bund_nii.affine
            header=bund_nii.header
            bund_data = bund_nii.get_fdata()

            tract60_index = np.array(list(set(list(range(72)))-set(bundles_index_12))).astype(int)
            print(tract60_index)
            bund_60 = bund_data[:,:,:,tract60_index].astype(np.float32)
            bundle_nii_60 = nib.Nifti1Image(bund_60, affine, header)
            nib.save(bundle_nii_60, os.path.join(dir, sub, 'bundle_masks_60.nii.gz'))



def extract_bundle_individual_hcp():
    dir = '/data4/wanliu/HCP_1.25mm_270/HCP_for_training_COPY'
    save_dir = '/data4/wanliu/HCP_1.25mm_270/HCP_for_training_COPY_SinTract'
    # used_sub=HCP_subjects
    used_sub = ["672756", "665254", "654754"]
    for sub in used_sub:
        os.makedirs(join(save_dir, sub))
        print(sub)
        all_bund_file = os.path.join(dir, sub, 'bundle_masks_72.nii.gz')
        bund_nii =nib.load(all_bund_file)
        affine=bund_nii.affine
        header=bund_nii.header
        bund_data = bund_nii.get_fdata()

        for ind, tract_name in enumerate(bundles):
            single_bund = bund_data[:,:,:,ind].astype(np.float32)
            bundle_nii = nib.Nifti1Image(single_bund, affine, header)
            nib.save(bundle_nii, os.path.join(save_dir, sub, tract_name+'.nii.gz'))


def extract_bundle_individual_hcp1():
    subjects_test=["802844", "792766", "792564", "789373", "786569", "784565", "782561", "779370", "771354", "770352",
               "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968", "673455", "672756",
               "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236", "620434", "613538"]
    bundles_novel_12 = ['CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'ILF_left', 'ILF_right','OR_left', 'OR_right', 'POPT_left', 'POPT_right',  'UF_left', 'UF_right']
    # bundles_index_new = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    dir = '/data4/wanliu/HCP_2.5mm_341k/HCP_for_training_COPY'
    save_dir = '/data4/wanliu/HCP_2.5mm_341k/HCP_for_training_COPY'
    for sub in subjects_test:
        save_path = join(save_dir, sub, 'single_tract_12')
        os.makedirs(save_path)
        print(sub)
        all_bund_file = os.path.join(dir, sub, 'bundle_masks_12.nii.gz')
        bund_nii =nib.load(all_bund_file)
        affine=bund_nii.affine
        header=bund_nii.header
        bund_data = bund_nii.get_fdata()

        for ind, tract_name in enumerate(bundles_novel_12):
            single_bund = bund_data[:,:,:,ind].astype(np.float32)
            bundle_nii = nib.Nifti1Image(single_bund, affine, header)
            nib.save(bundle_nii, os.path.join(save_path, tract_name+'.nii.gz'))


def extract_bundle_individual_hcp2():
    subjects_test=["802844", "792766", "792564", "789373", "786569", "784565", "782561", "779370", "771354", "770352",
               "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968", "673455", "672756",
               "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236", "620434", "613538"]
    bundles_novel_12 = ['CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'ILF_left', 'ILF_right','OR_left', 'OR_right', 'POPT_left', 'POPT_right',  'UF_left', 'UF_right']
    # bundles_index_new = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    dir = '/data7/wanliu/TractSeg_Run/one_shot_ensemble/Fewshot_SpottuneT5/HCP_2.5mm_34/Oneshot_845458_12/One_step_pretrain/0123_0.5to1'
    save_dir = '/data7/wanliu/TractSeg_Run/one_shot_ensemble/Fewshot_SpottuneT5/HCP_2.5mm_34/Oneshot_845458_12/One_step_pretrain/0123_0.5to1'
    for sub in subjects_test:
        save_path = join(save_dir,'single_tract',sub)
        os.makedirs(save_path)
        print(sub)
        all_bund_file = os.path.join(dir, sub+'.nii.gz')
        bund_nii =nib.load(all_bund_file)
        affine=bund_nii.affine
        header=bund_nii.header
        bund_data = bund_nii.get_fdata()

        for ind, tract_name in enumerate(bundles_novel_12):
            single_bund = bund_data[:,:,:,ind].astype(np.float32)
            bundle_nii = nib.Nifti1Image(single_bund, affine, header)
            nib.save(bundle_nii, os.path.join(save_path, tract_name+'.nii.gz'))


def extract_bundle_TT():
    ## for tt
    bundles_TT = ['CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right',  'UF_left', 'UF_right']
    bundles_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    subjects_TT = ['A00','A01','A02','A03','A04','A05','A06','A07','C00','C01','C02','C03','C04','C05','C06','C07','C08']

    ## extract 4 tracts
    bundles_TT_4 = ['CST_left', 'CST_right', 'OR_left', 'OR_right']
    bundles_index_4 = [0, 1, 4, 5]

    ## extract 6 tracts
    bundles_TT_6 = ['CST_left', 'CST_right','OR_left', 'OR_right', 'POPT_left', 'POPT_right']
    bundles_index_6 = [0, 1, 4, 5, 6, 7]

    dir = '/data4/wanliu/BT_1.7mm_270/HCP_for_training_COPY/second'
    for sub in os.listdir(dir):
    # for sub in subjects_TT:
        print(sub)
        all_bund_file = os.path.join(dir, sub, 'bundle_masks_10.nii.gz')
        bund_nii =nib.load(all_bund_file)
        affine=bund_nii.affine
        bund_data = bund_nii.get_fdata()

        novel_index_4 = np.array(bundles_index_4).astype(int)
        bund_new_4 = bund_data[:,:,:,novel_index_4].astype(np.uint8)
        bundle_nii_4 = nib.Nifti1Image(bund_new_4, affine)
        nib.save(bundle_nii_4, os.path.join(dir, sub, 'bundle_masks_4.nii.gz'))

        novel_index_6 = np.array(bundles_index_6).astype(int)
        bund_new_6 = bund_data[:,:,:,novel_index_6].astype(np.uint8)
        bundle_nii_6 = nib.Nifti1Image(bund_new_6, affine)
        nib.save(bundle_nii_6, os.path.join(dir, sub, 'bundle_masks_6.nii.gz'))


def extract_bundle_individual_tt1():
    subjects_test=['A00','A01','A02','A03','A04','A05','A06','A07','C00','C01','C02','C03','C04','C05','C06','C07','C08']
    bundles_novel_10 = ['CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right',  'UF_left', 'UF_right']
    # bundles_index_new = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    dir = '/data4/wanliu/BT_1.7mm_270/HCP_for_training_COPY'
    save_dir = '/data4/wanliu/BT_1.7mm_270/HCP_for_training_COPY'
    for sub in subjects_test:
        save_path = join(save_dir, sub, 'single_tract_10')
        os.makedirs(save_path)
        print(sub)
        all_bund_file = os.path.join(dir, sub, 'bundle_masks_10.nii.gz')
        bund_nii =nib.load(all_bund_file)
        affine=bund_nii.affine
        header=bund_nii.header
        bund_data = bund_nii.get_fdata()

        for ind, tract_name in enumerate(bundles_novel_10):
            single_bund = bund_data[:,:,:,ind].astype(np.float32)
            bundle_nii = nib.Nifti1Image(single_bund, affine, header)
            nib.save(bundle_nii, os.path.join(save_path, tract_name+'.nii.gz'))


def extract_bundle_individual_tt2():
    subjects_test=['A01','A02','A03','A04','A05','A06','A07','C00','C01','C02','C03','C04','C05','C06','C07','C08']
    bundles_novel_10 = ['CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right',  'UF_left', 'UF_right']
    # bundles_index_new = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    dir = '/data7/wanliu/TractSeg_Run/one_shot_ensemble/Fewshot_SpottuneT5/BT_1.7mm_270/Oneshot_A00_10/One_step_pretrain/0123_0.5to1'
    save_dir = '/data7/wanliu/TractSeg_Run/one_shot_ensemble/Fewshot_SpottuneT5/BT_1.7mm_270/Oneshot_A00_10/One_step_pretrain/0123_0.5to1'
    for sub in subjects_test:
        save_path = join(save_dir,'single_tract',sub)
        os.makedirs(save_path)
        print(sub)
        all_bund_file = os.path.join(dir, sub+'.nii.gz')
        bund_nii =nib.load(all_bund_file)
        affine=bund_nii.affine
        header=bund_nii.header
        bund_data = bund_nii.get_fdata()

        for ind, tract_name in enumerate(bundles_novel_10):
            single_bund = bund_data[:,:,:,ind].astype(np.float32)
            bundle_nii = nib.Nifti1Image(single_bund, affine, header)
            nib.save(bundle_nii, os.path.join(save_path, tract_name+'.nii.gz'))

## hcp 12 novel tract
# subjects_test=["992774", "991267", "987983", "984472", "983773", "979984", "978578", "965771", "965367", "959574",
#                "958976", "957974", "951457", "932554", "930449", "922854", "917255", "912447", "910241", "904044",
#                "901442", "901139", "901038", "899885", "898176", "896879", "896778", "894673", "889579", "877269",
#                "877168", "872764", "872158", "871964", "871762", "865363", "861456", "859671", "857263", "856766",
#                "849971", "837560", "814649", "765056", "761957", "759869", "756055", "753251", 
#                "751348", "749361", "748662", "748258", "742549",
#                "734045", "732243", "729557", "729254", "715647", "715041", "709551", "705341", "704238", "702133"]

# subjects_test=["802844", "792766", "792564", "789373", "786569", "784565", "782561", "779370", "771354", "770352",
#                "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968", "673455", "672756",
#                "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236", "620434", "613538"]
# bundles_novel_12 = ['CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'ILF_left', 'ILF_right','OR_left', 'OR_right', 'POPT_left', 'POPT_right',  'UF_left', 'UF_right']
# bundles_index_new = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# ## extract 4 tracts
# bundles_novel_4 = ['CST_left', 'CST_right', 'OR_left', 'OR_right']
# bundles_index_new_4 = [0, 1, 6, 7]
# ## extract 6 tracts
# bundles_novel_6 = ['CST_left', 'CST_right','OR_left', 'OR_right', 'POPT_left', 'POPT_right']
# bundles_index_new_6 = [0, 1, 6, 7, 8, 9]

subjects_test=['A01','A02','A03','A04','A05','A06','A07','C00','C01','C02','C03','C04','C05','C06','C07','C08']
bundles_novel_10 = ['CST_left', 'CST_right', 'FPT_left', 'FPT_right','OR_left', 'OR_right', 'POPT_left', 'POPT_right',  'UF_left', 'UF_right']
bundles_index_new = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
## extract 4 tracts
bundles_novel_4 = ['CST_left', 'CST_right', 'OR_left', 'OR_right']
bundles_index_new_4 = [0, 1, 4, 5]
## extract 6 tracts
bundles_novel_6 = ['CST_left', 'CST_right','OR_left', 'OR_right', 'POPT_left', 'POPT_right']
bundles_index_new_6 = [0, 1, 4, 5, 6, 7]


def extract_bundle_atlas():
    # dir = '/data4/wanliu/HCP_2.5mm_341k/HCP_AtlasFSL/Oneshot_845458_results'
    dir = '/data4/wanliu/BT_1.7mm_270/HCP_AtlasFSL/Oneshot_A00_results'

    # sub_list = os.listdir(dir)
    # print(bundles_TT)
    for sub in subjects_test:
        all_bund_file = os.path.join(dir, sub, 'bundle_masks_new10_atlasfsl.nii.gz')
        if os.path.exists(all_bund_file)==True:
            print(sub)
            bund_nii =nib.load(all_bund_file)
            affine=bund_nii.affine
            bund_data = bund_nii.get_fdata()

            print(bundles_index_new_6)
            novel_index_6 = np.array(bundles_index_new_6).astype(int)
            bund_6 = bund_data[:,:,:,novel_index_6].astype(np.float32)
            bundle_nii = nib.Nifti1Image(bund_6, affine)
            nib.save(bundle_nii, os.path.join(dir, sub, 'bundle_masks_new6_atlasfsl.nii.gz'))
            
            print(bundles_index_new_4)
            novel_index_4 = np.array(bundles_index_new_4).astype(int)
            bund_4 = bund_data[:,:,:,novel_index_4].astype(np.float32)
            bundle_nii = nib.Nifti1Image(bund_4, affine)
            nib.save(bundle_nii, os.path.join(dir, sub, 'bundle_masks_new4_atlasfsl.nii.gz'))


def combine_wm_annotation():
    save_dir='/data4/wanliu/BT_1.7mm_270/HCP_for_training_COPY/second'
    # init_dir = '/data4/wanliu/BT_1.7mm_initdata/BT_1.7mm_270/second_data'
    bundles=['cst_left','cst_right','fpt_left','fpt_right','or_left','or_right','popt_left','popt_right','uf_left','uf_right']
    dir = '/data4/wanliu/BT_annotation/HC/annotation/second/binary'
    for sub in os.listdir(dir):
        if os.path.exists(join(save_dir, sub)) is False:
            os.makedirs(join(save_dir, sub))
        mask_path = join(dir, sub, sub+'_cst_left.nii.gz')
        affine = nib.load(mask_path).affine
        shape = nib.load(mask_path).get_fdata().shape
        print([shape[0],shape[1],shape[2],10])
        allmask_data = np.zeros([shape[0],shape[1],shape[2],len(bundles)])
        for i, tract in enumerate(bundles):
            tract_path = join(dir, sub, sub+'_'+tract+'.nii.gz')
            print(tract_path)
            allmask_data[:,:,:,i]=nib.load(tract_path).get_fdata()
        nib.save(nib.Nifti1Image(allmask_data, affine), join(save_dir, sub, 'bundle_masks_'+str(len(bundles))+'.nii.gz'))








if __name__ == "__main__":
    extract_bundle_hcp()
    # extract_bundle_hcp_12()
    # extract_bundle_TT()
    # extract_bundle_atlas()
    # combine_wm_annotation()
    # extract_bundle_individual_hcp2()
    # extract_bundle_individual_tt2()