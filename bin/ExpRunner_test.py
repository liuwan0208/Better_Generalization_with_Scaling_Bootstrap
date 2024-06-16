#!/usr/bin/env python

"""
This module is for training the model. See Readme.md for more details about training your own model.

Examples:
    Run local:
    $ ExpRunner --config=XXX

    Predicting with new config setup:
    $ ExpRunner --train=False --test=True --lw --config=XXX
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import os
import importlib
import argparse
import pickle as pkl
from pprint import pprint
import distutils.util
from os.path import join
import os

import nibabel as nib
import numpy as np
import time
import torch

from tractseg.libs import data_utils
from tractseg.libs import direction_merger
from tractseg.libs import exp_utils
from tractseg.libs import img_utils
from tractseg.libs import peak_utils
from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import trainer
from tractseg.data.data_loader_training import DataLoaderTraining as DataLoaderTraining2D
from tractseg.data.data_loader_training_3D import DataLoaderTraining as DataLoaderTraining3D
from tractseg.data.data_loader_inference import DataLoaderInference
from tractseg.data import dataset_specific_utils
from tractseg.models.base_model import BaseModel

# from bin.utils_add import compute_dice_score

warnings.simplefilter("ignore", UserWarning)  # hide scipy warnings
warnings.simplefilter("ignore", FutureWarning)  # hide h5py warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")  # hide Cython benign warning
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")  # hide Cython benign warning


def compute_dice_score(predict, gt):
    overlap = 2.0 * np.sum(predict * gt)
    return overlap / (np.sum(predict) + np.sum(gt))


def compute_iou_score(predict, gt):
    overlap = np.sum(predict * gt)
    return overlap / (np.sum(predict) + np.sum(gt) - overlap)


def compute_rvd_score(predict, gt):
    rvd = abs(1 - np.sum(predict) / np.sum(gt))
    return rvd


def main():
    parser = argparse.ArgumentParser(description="Train a network on your own data to segment white matter bundles.",
                                     epilog="Written by Jakob Wasserthal. Please reference 'Wasserthal et al. "
                                            "TractSeg - Fast and accurate white matter tract segmentation. "
                                            "https://doi.org/10.1016/j.neuroimage.2018.07.070)'")
    parser.add_argument("--config", metavar="name", help="Name of configuration to use",
                        default='my_custom_experiment')
    parser.add_argument("--train", metavar="True/False", help="Train network",
                        type=distutils.util.strtobool, default=False)
    parser.add_argument("--test", metavar="True/False", help="Test network",
                        type=distutils.util.strtobool, default=True)
    parser.add_argument("--seg", action="store_true", help="Create binary segmentation", default=True)
    parser.add_argument("--probs", action="store_true", help="Create probmap segmentation")
    parser.add_argument("--lw", action="store_true", help="Load weights of pretrained net")
    parser.add_argument("--only_val", action="store_true", help="only run validation")
    parser.add_argument("--en", metavar="name", help="Experiment name")
    parser.add_argument("--fold", metavar="N", help="Which fold to train when doing CrossValidation", type=int)
    parser.add_argument("--verbose", action="store_true", help="Show more intermediate output", default=True)
    args = parser.parse_args()

    Config = getattr(importlib.import_module("tractseg.experiments.base"), "Config")()
    if args.config:
        # Config.__dict__ does not work properly therefore use this approach
        Config = getattr(importlib.import_module("tractseg.experiments.custom." + args.config), "Config")()

    if args.en:
        Config.EXP_NAME = args.en  ## my_custom_experiment

    Config.TRAIN = bool(args.train)  ## True
    Config.TEST = bool(args.test)  ## False
    Config.SEGMENT = args.seg  ## False
    if args.probs:
        Config.GET_PROBS = True  ## False
    if args.lw:
        Config.LOAD_WEIGHTS = args.lw  ## False
    if args.fold:
        Config.CV_FOLD = args.fold  ## 0
    if args.only_val:
        Config.ONLY_VAL = True  ## False
    Config.VERBOSE = args.verbose  ## True:show more intermedia output

    Config.MULTI_PARENT_PATH = join(C.EXP_PATH, Config.EXP_MULTI_NAME)  ## '//home/wanliu/TractSeg/hcp_exp/'
    Config.EXP_PATH = join(C.EXP_PATH, Config.EXP_MULTI_NAME,
                           Config.EXP_NAME)  ## '/home/wanliu/TractSeg/hcp_exp/my_custom_experiment'

    ##### modify subject numbers
    Config.TRAIN_SUBJECTS, Config.VALIDATE_SUBJECTS, Config.TEST_SUBJECTS = dataset_specific_utils.get_cv_fold(
        Config.CV_FOLD,
        dataset=Config.DATASET)
    Config.TRAIN_SUBJECTS = Config.TRAIN_SUBJECTS[0:20]

    if Config.WEIGHTS_PATH == "":
        Config.WEIGHTS_PATH = exp_utils.get_best_weights_path(Config.EXP_PATH, Config.LOAD_WEIGHTS)

    # Autoset input dimensions based on settings
    Config.INPUT_DIM = dataset_specific_utils.get_correct_input_dim(Config)  ## (144,144)
    Config = dataset_specific_utils.get_labels_filename(Config)  ## 'LABELS_FILENAME': 'bundle_masks_72'

    if Config.EXPERIMENT_TYPE == "peak_regression":
        Config.NR_OF_CLASSES = 3 * len(dataset_specific_utils.get_bundle_names(Config.CLASSES)[1:])
    else:
        Config.NR_OF_CLASSES = len(dataset_specific_utils.get_bundle_names(Config.CLASSES)[1:])  ## 72

    if Config.TRAIN and not Config.ONLY_VAL:
        Config.EXP_PATH = exp_utils.create_experiment_folder(Config.EXP_NAME, Config.MULTI_PARENT_PATH, Config.TRAIN)

    if Config.TRAIN:
        if Config.VERBOSE:
            print("Hyperparameters:")
            exp_utils.print_Configs(Config)

        with open(join(Config.EXP_PATH, "Hyperparameters.txt"), "a") as f:
            Config_dict = {attr: getattr(Config, attr) for attr in dir(Config)
                           if not callable(getattr(Config, attr)) and not attr.startswith("__")}
            pprint(Config_dict, f)
        if Config.DIM == "2D":
            data_loader = DataLoaderTraining2D(Config)
        else:
            data_loader = DataLoaderTraining3D(Config)
        print("Training...")
        model = BaseModel(Config)
        trainer.train_model(Config, model, data_loader)
    Config = exp_utils.get_correct_labels_type(Config)

    test_sub = ["802844", "792766", "792564", "789373", "786569", "784565", "782561", "779370", "771354", "770352",
               "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968", "673455", "672756",
              "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236", "620434", "613538"]
    # test_sub = ["802844"]

    print('num of test subs', len(test_sub))

    Config.TEST_SUBJECTS = test_sub
    tract_num = Config.NR_OF_CLASSES

    if Config.TEST:
        Config.MODEL = 'UNet_SelfAtt'
        Config.NORMALIZE_PER_CHANNEL = False
        
        Config.WEIGHTS_PATH = '/data7/wanliu/TractSeg_Run/hcp_exp7/my_custom_experiment_x88/best_weights_ep110.npz'
        save_path = '/data7/wanliu/TractSeg_Run/hcp_exp7/my_custom_experiment_x88'

        Test_peak_dir_list = ['/data4/wanliu/HCP_1.25mm_270/HCP_for_training_COPY',
                              '/data4/wanliu/HCP_1.25mm_901k_18b0/HCP_for_training_COPY',
                              '/data4/wanliu/HCP_1.25mm_121k_18b0/HCP_for_training_COPY',
                              '/data4/wanliu/HCP_1.25mm_341k/HCP_for_training_COPY',
                              '/data4/wanliu/HCP_1.25mm_361k2k/HCP_for_training_COPY',]
        npy_save_file_list = ['Dice_test_HCP_1.25mm_270.npy',
                              'Dice_test_HCP_1.25mm_901k.npy',
                              'Dice_test_HCP_1.25mm_121k.npy',
                              'Dice_test_HCP_1.25mm_341k.npy',
                              'Dice_test_HCP_1.25mm_361k2k.npy',]
        logo_list = ['HCP_1.25mm_270','HCP_1.25mm_901k','HCP_1.25mm_121k','HCP_1.25mm_341k','HCP_1.25mm_361k2k']

        for index in range(len(Test_peak_dir_list)):
            Test_peak_dir = Test_peak_dir_list[index]
            npy_save_file = npy_save_file_list[index]
            logo = logo_list[index]

            inference = True
            cal_metric = True

            test_epoch = Config.WEIGHTS_PATH.split('/')[-1].split('_')[-1].split('.')[0]
            print("Loading weights in ", Config.WEIGHTS_PATH)
            model = BaseModel(Config, inference=True)
            model.load_model(Config.WEIGHTS_PATH)

            if inference:
                for subject in Config.TEST_SUBJECTS:
                    print("Get_segmentation subject {}".format(subject))
                    peak_path = join(Test_peak_dir, subject, 'mrtrix_peaks.nii.gz')
                    data_img = nib.load(peak_path)
                    data_affine = data_img.affine
                    data0 = data_img.get_fdata()
                    data = np.nan_to_num(data0)
                    data, _, bbox, original_shape = data_utils.crop_to_nonzero(data)

                    data, transformation = data_utils.pad_and_scale_img_to_square_img(data, target_size=Config.INPUT_DIM[0],
                                                                                    nr_cpus=-1)
                    seg_xyz, _ = direction_merger.get_seg_single_img_3_directions(Config, model, data=data,
                                                                                    scale_to_world_shape=False,
                                                                                    only_prediction=True,
                                                                                    batch_size=1)

                    seg = direction_merger.mean_fusion(Config.THRESHOLD, seg_xyz, probs=False)
                    seg = data_utils.cut_and_scale_img_back_to_original_img(seg, transformation, nr_cpus=-1)
                    seg = data_utils.add_original_zero_padding_again(seg, bbox, original_shape, Config.NR_OF_CLASSES)


                    print('save segmentation results')
                    img_seg = nib.Nifti1Image(seg.astype(np.uint8), data_affine)
                    output_all_bund = join(save_path, "segmentation_"+ logo + '_' + test_epoch, 'all_bund_seg')
                    exp_utils.make_dir(output_all_bund)
                    print(output_all_bund)
                    nib.save(img_seg, join(output_all_bund, subject + ".nii.gz"))

                    bundles = dataset_specific_utils.get_bundle_names(Config.CLASSES)[1:]
                    output_indiv_bund = join(save_path, "segmentation_"+ logo + '_' + test_epoch, 'indiv_bund_seg', subject)
                    exp_utils.make_dir(output_indiv_bund)
                    print(output_indiv_bund)
                    for idx, bundle in enumerate(bundles):
                        img_seg = nib.Nifti1Image(seg[:, :, :, idx], data_affine)
                        nib.save(img_seg, join(output_indiv_bund, bundle + ".nii.gz"))


            ## -----------------------Compute mean Dice of each tract------------------------
            if cal_metric:
                print('computing dice coeff')
                all_subjects = Config.TEST_SUBJECTS
                Seg_path = join(save_path, "segmentation_"+ logo + '_' + test_epoch, 'all_bund_seg')
                Label_path = Test_peak_dir

                Dice_all = np.zeros([len(all_subjects), tract_num])
                IOU_all = np.zeros([len(all_subjects), tract_num])
                RVD_all = np.zeros([len(all_subjects), tract_num]) 
                print('Dice_all_shape', Dice_all.shape)

                for subject_index in range(len(all_subjects)):
                    subject = all_subjects[subject_index]
                    print("Get_test subject {}".format(subject))
                    seg_path = join(Seg_path, subject + ".nii.gz")
                    label_path = join(Label_path, subject, Config.LABELS_FILENAME + '.nii.gz')
                    print(seg_path)
                    print(label_path)
                    seg = nib.load(seg_path).get_fdata()
                    label= nib.load(label_path).get_fdata()
                    

                    for tract_index in range(label.shape[-1]):
                        Dice_all[subject_index, tract_index] = compute_dice_score(seg[:,:,:,tract_index], label[:,:,:,tract_index])
                        IOU_all[subject_index, tract_index] = compute_iou_score(seg[:,:,:,tract_index], label[:,:,:,tract_index])
                        RVD_all[subject_index, tract_index] = compute_rvd_score(seg[:,:,:,tract_index], label[:,:,:,tract_index])

                    with open(join(save_path, "test_dice_"+ logo + '_' +str(test_epoch)+".txt"), 'a') as f:
                        f.write('Dice of subject {} is \n {} \n'.format(subject, Dice_all[subject_index, :]))
                    with open(join(save_path, "test_iou_"+ logo + '_' +str(test_epoch)+".txt"), 'a') as f:
                        f.write('IOU of subject {} is \n {} \n'.format(subject, IOU_all[subject_index, :]))
                    with open(join(save_path, "test_rvd_"+ logo + '_' +str(test_epoch)+".txt"), 'a') as f:
                        f.write('RVD of subject {} is \n {} \n'.format(subject, RVD_all[subject_index, :]))

                np.save(join(save_path,npy_save_file), Dice_all)


                Dice_mean = np.mean(Dice_all, 0)
                Dice_average = np.mean(Dice_all)
                IOU_mean = np.mean(IOU_all, 0)
                IOU_average = np.mean(IOU_all)
                RVD_mean = np.mean(RVD_all, 0)
                RVD_average = np.mean(RVD_all)

                bundles = dataset_specific_utils.get_bundle_names(Config.CLASSES)[1:]
                print(len(bundles))

                with open(join(save_path, "test_dice_"+ logo + '_' +str(test_epoch)+".txt"),'a') as f:
                    for index in range(tract_num):
                        log = '{}: {} \n'.format(bundles[index], Dice_mean[index])
                        f.write(log)
                    log = 'mean dice of all tract is:{}\n'.format(Dice_average)
                    f.write(log)
                    print(log)

                with open(join(save_path, "test_iou_"+ logo + '_' +str(test_epoch)+".txt"),'a') as f:
                    for index in range(tract_num):
                        log = '{}: {} \n'.format(bundles[index], IOU_mean[index])
                        f.write(log)
                    log = 'mean iou of all tract is:{}\n'.format(IOU_average)
                    f.write(log)
                    print(log)

                with open(join(save_path, "test_rvd_"+ logo + '_' +str(test_epoch)+".txt"),'a') as f:
                    for index in range(tract_num):
                        log = '{}: {} \n'.format(bundles[index], RVD_mean[index])
                        f.write(log)
                    log = 'mean rvd of all tract is:{}\n'.format(RVD_average)
                    f.write(log)
                    print(log)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()

