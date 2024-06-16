om dipy.reconst.shore import ShoreModel, shore_matrix
# from dipy.viz import window, actor
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
import numpy as np
from dipy.reconst.cache import Cache
import nibabel as nib
from os.path import join
from random import sample,choices
import shutil
import os


def select_woreplace(in_signal): #[channel]
    select_index = sample(range(0, in_signal.shape[0]), in_signal.shape[0]) ## this sample index without replacement
    select_mat = np.zeros([in_signal.shape[0], in_signal.shape[0]])
    for (i, index) in enumerate(select_index):
        select_mat[i, index] = 1
    out_signal = np.dot(select_mat, in_signal)
    return out_signal

def select_withreplace(in_signal): #[channel]
    select_index = choices(range(0, in_signal.shape[0]), k=in_signal.shape[0]) ## this sample index without replacement
    # print(select_index)
    select_mat = np.zeros([in_signal.shape[0], in_signal.shape[0]])
    for (i, index) in enumerate(select_index):
        select_mat[i, index] = 1
    out_signal = np.dot(select_mat, in_signal)
    return out_signal

def bootstrap_image(res):
    res_reshape = np.reshape(res, [-1, res.shape[-1]]) #[N, channel]
    res_bootstrap = np.zeros(res_reshape.shape) #[N, channel]
    for i in range(res_reshape.shape[0]): #[N]
        res_bootstrap[i,:] = select_withreplace(res_reshape[i,:])
    res_bootstrap = np.reshape(res_bootstrap, res.shape)
    return res_bootstrap



image_bootstrap = True
dti_process = True
split_peaks = True
init_data_process = False #whether there is brain mask of the initial HCP subject; if not, than the init_data_process is performed to generate the brain mask

single_shell = False ## whether single b value

subjects = ["992774", "991267", "987983", "984472", "983773", "979984", "978578", "965771", "965367", "959574",#train1-30
            "958976", "957974", "951457", "932554", "930449", "922854", "917255", "912447", "910241", "904044",
            "901442", "901139", "901038", "899885", "898176", "896879", "896778", "894673", "889579", "877269",

            "877168", "872764", "872158", "871964", "871762", "865363", "861456", "859671", "857263", "856766",#train31-50
            "849971", "845458", "837964", "837560", "833249", "833148", "826454", "826353", "816653", "814649",

            "802844", "792766", "792564", "789373", "786569", "784565", "782561", "779370", "771354", "770352", #test

            "765056", "761957", "759869", "756055", "753251", # train 50-55

            "751348", "749361", "748662", "748258", "742549",
            "734045", "732243", "729557", "729254", "715647", "715041", "709551", "705341", "704238", "702133",#val

            "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968", "673455", "672756",#test
            "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236", "620434", "613538"]##100

##*******  load data *********
training_sub = ["992774", "991267", "987983", "984472", "983773", "979984", "978578", "965771", "965367", "959574",
             "958976", "957974", "951457", "932554", "930449", "922854", "917255", "912447", "910241", "904044",
             "901442", "901139", "901038", "899885", "898176", "896879", "896778", "894673", "889579", "877269",
             "877168", "872764", "872158", "871964", "871762", "865363", "861456", "859671", "857263", "856766",
             "849971", "845458", "837964", "837560", "833249", "833148", "826454", "826353", "816653", "814649",
             "765056", "761957", "759869", "756055", "753251"]# train 55 subjects in total
             #                 "877168", "872764", "872158", "871964", "871762", "865363", "861456", "859671", "857263", "856766",
             #                 "849971", "837560", "814649", "765056", "761957", ##train 45
             # 				"751348", "749361", "748662", "748258", "742549",
             #                 "734045", "732243", "729557", "729254", "715647", "715041", "709551", "705341", "704238", "702133"]##val 15


train_sub = training_sub[0:55]
print('train_sub', train_sub)


for sub in train_sub:
    print(sub)


    dti_path = join('/data7/wanliu/Data/HCP_initdata/HCP_1.25mm_270', sub)
    save_path = join('/data7/wanliu/Data_Residual_Bootstrap/HCP_initdata_bootstrap/HCP_1.25mm_270', sub)
    radial_order = 6 #decide how many basis are used to represent the signal

    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    ratio_list=[2,3,4]

    fraw  = join(dti_path,'data.nii.gz')
    fbval = join(dti_path,'bvals')
    fbvec = join(dti_path,'bvecs')
    data, affine = load_nifti(fraw)
    gtab = gradient_table(bvals=fbval, bvecs=fbvec)
    # shutil.copy(fraw, join(save_path,'data.nii.gz'))
    # shutil.copy(fbval, join(save_path,'bvals'))
    # shutil.copy(fbvec, join(save_path,'bvecs'))
    # print('copy initial data done:', sub)


    if image_bootstrap == True:
        ##******* seperate b0 and non-b0 image and gradient *******
        bvals = list(gtab.bvals)
        index_b0 = np.array([i for (i,v) in enumerate(bvals) if v<10]).astype(int)
        bvec_b0  = gtab.bvecs[index_b0]
        bval_b0  = gtab.bvals[index_b0]
        data_b0  = data[:,:,:,index_b0]

        index_Nb0 = np.array([i for (i,v) in enumerate(bvals) if v>=900 and v<=4000]).astype(int)
        bvec_Nb0  = gtab.bvecs[index_Nb0]
        bval_Nb0  = gtab.bvals[index_Nb0]
        data_Nb0  = data[:,:,:,index_Nb0]

        ##******* recover b0 with mean-b0 *******
        mean_b0 = np.mean(data_b0, axis=-1)
        mean_b0_extension = np.repeat(np.expand_dims(mean_b0, axis=-1), repeats=data_b0.shape[-1], axis=-1)
        residual_b0 = data_b0-mean_b0_extension



        ##******* recover data_non-b0 with shore basis *******
        gtab_Nb0 = gradient_table(bval_Nb0, bvec_Nb0)
        zeta = 700
        tau_default = 1. / (4 * np.pi ** 2) #1/(2pi)^2
        if (gtab_Nb0.big_delta is None) or (gtab_Nb0.small_delta is None):
            tau = tau_default
        else:
            tau = gtab_Nb0.big_delta - gtab_Nb0.small_delta / 3.0

        M = shore_matrix(radial_order, zeta, gtab_Nb0, tau) #[270,50]
        MpseudoInv = np.dot(np.linalg.inv(np.dot(M.T, M)), M.T) #[50,270]
        H = np.dot(M, MpseudoInv) #[270,270]
        # print(np.diag(H))

        data_Nb0_reshape = np.reshape(data_Nb0,[-1,data_Nb0.shape[-1]]) #[145*174*145, 270]
        rec_data_Nb0 = np.dot(H, data_Nb0_reshape.T) #[270,50]*[50,N]=[270,N]
        data_Nb0_recovery = np.reshape(rec_data_Nb0.T, data_Nb0.shape).astype(np.float32)

        residual_Nb0 = data_Nb0-data_Nb0_recovery #[145,174,145,270]
        cor_coef_Nb0 = 1/(np.sqrt(1-np.diag(H))) #[270]---1/(sqrt(1-hii))
        residual_correct_Nb0 = residual_Nb0 * cor_coef_Nb0 #[145,174,145,270]---e^m

        for ratio in ratio_list:
            ##******* random select residual with replacement for each voxel *********
            residual_Nb0_bootstrap = bootstrap_image(residual_correct_Nb0)
            data_new_Nb0 = data_Nb0_recovery + ratio * residual_Nb0_bootstrap
            residual_b0_bootstrap  = bootstrap_image(residual_b0)
            data_new_b0  = mean_b0_extension + ratio * residual_b0_bootstrap

            ##******* combine the new data of b0 and non-b0 *********
            data_new = np.zeros(data.shape)
            data_new[:,:,:,index_b0] = data_new_b0
            data_new[:,:,:,index_Nb0] = data_new_Nb0

            ##******* image save *******
            nib.save(nib.Nifti1Image(data_new.astype(np.float32), affine), join(save_path, 'new_data_ratio'+str(ratio)+'.nii.gz'))
            print('new image generate done:', ratio)


    if dti_process == True:
        bvecs_path = join(save_path, 'bvecs')
        bvals_path = join(save_path, 'bvals')
        if init_data_process==True:
            data_path = join(save_path, 'data.nii.gz')
            dti_fit_path = join(save_path, 'dti_fit')
            brainmask_path = join(save_path, 'brain_mask.nii.gz')
            bvecs_path = join(save_path, 'bvecs')
            bvals_path = join(save_path, 'bvals')
            ##-------------extract brain mask-------------
            # os.system("bet " + join(save_path, 'data') + " " +join(save_path, "brain.nii.gz") + " -f 0.3 -g 0 -m")
            # os.system("rm "  + save_path + "/brain.nii.gz")
            #-----------------dti fit, focus on data_v1 and visualize in fsl ---------------
            if os.path.exists(dti_fit_path) is False:
                os.makedirs(dti_fit_path)
            # os.system("dtifit --data=" + data_path + " --out=" + dti_fit_path + "/data --mask=" + brainmask_path + " --bvecs=" + bvecs_path + " --bvals=" + bvals_path)
            if single_shell == False:
                os.system("dwi2response dhollander -mask "+brainmask_path + " " + data_path + " "+join(dti_fit_path,'RF_WM.txt')+" "+join(dti_fit_path,'RF_GM.txt')+" "+join(dti_fit_path,'RF_CSF.txt')\
                            +' -fslgrad '+bvecs_path+" "+bvals_path+" -mask "+brainmask_path+""+' -force')
                os.system("dwi2fod msmt_csd " + data_path + " " + join(dti_fit_path, 'RF_WM.txt') + " " + join(dti_fit_path, 'fod_wm.nii.gz') +
                          " " + join(dti_fit_path, 'RF_GM.txt') + " " + join(dti_fit_path, 'fod_gm.nii.gz') +
                          " " + join(dti_fit_path, 'RF_CSF.txt') + " " + join(dti_fit_path, 'fod_csf.nii.gz') +
                          ' -fslgrad ' + bvecs_path + " " + bvals_path + " -mask " + brainmask_path+""+' -force')
            else:
                os.system("dwi2response tournier "+data_path+" "+join(dti_fit_path,'response.txt')\
                            +' -fslgrad '+bvecs_path+" "+bvals_path+" -mask "+brainmask_path+' -force')
                os.system("dwi2fod csd "+data_path+" "+join(dti_fit_path,'response.txt')+" "+join(dti_fit_path,'fod_wm.nii.gz')+' -fslgrad '+bvecs_path+" "+bvals_path+" -mask "+brainmask_path+" -quiet" + ""+' -force')
            os.system("sh2peaks " +join(dti_fit_path,'fod_wm.nii.gz')+' '+join(dti_fit_path,"peaks.nii.gz")+' -force'+" -quiet" + "")

        for ratio in ratio_list:
            data_path = join(save_path, 'new_data_ratio'+str(ratio)+'.nii.gz')
            dti_fit_path = join(save_path, 'dti_fit_ratio'+str(ratio))
            # brainmask_path = join(save_path, 'ratio'+str(ratio)+'_brain_mask.nii.gz')
            brainmask_path = join(save_path, 'brain_mask.nii.gz')
            ##------------- extract brain mask -------------
            # os.system("bet " + join(save_path, 'new_data_ratio'+str(ratio)) + " " +join(save_path, 'ratio'+str(ratio)+'_brain.nii.gz') + " -f 0.1 -g 0 -m")
            # os.system("rm "  + join(save_path, 'ratio'+str(ratio)+'_brain.nii.gz'))
            ##----------------- dti fit, focus on data_v1 and visualize in fsl ---------------
            if os.path.exists(dti_fit_path) is False:
                os.makedirs(dti_fit_path)
            # os.system("dtifit --data="+data_path+" --out="+dti_fit_path+"/data --mask="+brainmask_path+" --bvecs="+bvecs_path+" --bvals="+bvals_path)
            if single_shell == False:
                os.system("dwi2response dhollander -mask "+brainmask_path + " " + data_path + " "+join(dti_fit_path,'RF_WM.txt')+" "+join(dti_fit_path,'RF_GM.txt')+" "+join(dti_fit_path,'RF_CSF.txt')\
                            +' -fslgrad '+bvecs_path+" "+bvals_path+" -mask "+brainmask_path+""+' -force')
                os.system("dwi2fod msmt_csd " + data_path + " " + join(dti_fit_path, 'RF_WM.txt') + " " + join(dti_fit_path, 'fod_wm.nii.gz') +
                          " " + join(dti_fit_path, 'RF_GM.txt') + " " + join(dti_fit_path, 'fod_gm.nii.gz') +
                          " " + join(dti_fit_path, 'RF_CSF.txt') + " " + join(dti_fit_path, 'fod_csf.nii.gz') +
                          ' -fslgrad ' + bvecs_path + " " + bvals_path + " -mask " + brainmask_path+""+' -force')
            else:
                os.system("dwi2response tournier "+data_path+" "+join(dti_fit_path,'response.txt')\
                            +' -fslgrad '+bvecs_path+" "+bvals_path+" -mask "+brainmask_path+' -force')
                os.system("dwi2fod csd "+data_path+" "+join(dti_fit_path,'response.txt')+" "+join(dti_fit_path,'fod_wm.nii.gz')+' -fslgrad '+bvecs_path+" "+bvals_path+" -mask "+brainmask_path+" -quiet" + ""+' -force')
            os.system("sh2peaks " +join(dti_fit_path,'fod_wm.nii.gz')+' '+join(dti_fit_path,"peaks.nii.gz")+' -force'+" -quiet" + "")


    if split_peaks==True:
        if init_data_process==True:
            dti_fit_path = join(save_path, 'dti_fit')
            peaks_path = join(dti_fit_path, 'peaks.nii.gz')
            peaks_nii=nib.load(peaks_path)
            affine = peaks_nii.affine
            peaks = peaks_nii.get_fdata()

            direction1_index = np.array([0, 1, 2]).astype(int)
            direction2_index = np.array([3, 4, 5]).astype(int)
            direction3_index = np.array([6, 7, 8]).astype(int)
            peaks_direction1 = peaks[:,:,:,direction1_index]
            peaks_direction2 = peaks[:,:,:,direction2_index]
            peaks_direction3 = peaks[:,:,:,direction3_index]
            # nib.save(nib.Nifti1Image(peaks_direction1.astype(np.float32),affine),join(dti_fit_path,'peaks_direction1.nii.gz'))
            # nib.save(nib.Nifti1Image(peaks_direction2.astype(np.float32),affine),join(dti_fit_path,'peaks_direction2.nii.gz'))
            # nib.save(nib.Nifti1Image(peaks_direction3.astype(np.float32),affine),join(dti_fit_path,'peaks_direction3.nii.gz'))

            ##********* change x-axis (the first channel in the 3-dim vector) to the inverse direction********
            peaks_direction1[:,:,:,0] = -peaks_direction1[:,:,:,0]
            peaks_direction2[:,:,:,0] = -peaks_direction2[:,:,:,0]
            peaks_direction3[:,:,:,0] = -peaks_direction3[:,:,:,0]
            nib.save(nib.Nifti1Image(peaks_direction1.astype(np.float32),affine),join(dti_fit_path,'peaks_direction1_xinverse.nii.gz'))
            nib.save(nib.Nifti1Image(peaks_direction2.astype(np.float32),affine),join(dti_fit_path,'peaks_direction2_xinverse.nii.gz'))
            nib.save(nib.Nifti1Image(peaks_direction3.astype(np.float32),affine),join(dti_fit_path,'peaks_direction3_xinverse.nii.gz'))


        for ratio in ratio_list:
            dti_fit_path = join(save_path, 'dti_fit_ratio'+str(ratio))
            peaks_path = join(dti_fit_path, 'peaks.nii.gz')
            peaks_nii = nib.load(peaks_path)
            peaks = peaks_nii.get_fdata()

            direction1_index = np.array([0, 1, 2]).astype(int)
            direction2_index = np.array([3, 4, 5]).astype(int)
            direction3_index = np.array([6, 7, 8]).astype(int)
            peaks_direction1 = peaks[:,:,:,direction1_index]
            peaks_direction2 = peaks[:,:,:,direction2_index]
            peaks_direction3 = peaks[:,:,:,direction3_index]
            # nib.save(nib.Nifti1Image(peaks_direction1.astype(np.float32),affine),join(dti_fit_path,'peaks_direction1.nii.gz'))
            # nib.save(nib.Nifti1Image(peaks_direction2.astype(np.float32),affine),join(dti_fit_path,'peaks_direction2.nii.gz'))
            # nib.save(nib.Nifti1Image(peaks_direction3.astype(np.float32),affine),join(dti_fit_path,'peaks_direction3.nii.gz'))

            ##********* change x-axis (the first channel in the 3-dim vector) to the inverse direction********
            peaks_direction1[:,:,:,0] = -peaks_direction1[:,:,:,0]
            peaks_direction2[:,:,:,0] = -peaks_direction2[:,:,:,0]
            peaks_direction3[:,:,:,0] = -peaks_direction3[:,:,:,0]
            nib.save(nib.Nifti1Image(peaks_direction1.astype(np.float32),affine),join(dti_fit_path,'peaks_direction1_xinverse.nii.gz'))
            nib.save(nib.Nifti1Image(peaks_direction2.astype(np.float32),affine),join(dti_fit_path,'peaks_direction2_xinverse.nii.gz'))
            nib.save(nib.Nifti1Image(peaks_direction3.astype(np.float32),affine),join(dti_fit_path,'peaks_direction3_xinverse.nii.gz'))

