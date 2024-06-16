
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
from os.path import join
import importlib
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adamax
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from timm.scheduler.plateau_lr import PlateauLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler

try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    pass

from tractseg.libs import pytorch_utils
from tractseg.libs import exp_utils
from tractseg.libs import metric_utils
from thop import profile

class BaseModel:
    def __init__(self, Config, inference=False):
        self.Config = Config

        # Do not use during inference because uses a lot more memory
        if not inference:
            torch.backends.cudnn.benchmark = True

        if self.Config.NR_CPUS > 0:
            torch.set_num_threads(self.Config.NR_CPUS)

        if self.Config.SEG_INPUT == "Peaks" and self.Config.TYPE == "single_direction":#true
            NR_OF_GRADIENTS = self.Config.NR_OF_GRADIENTS
        elif self.Config.SEG_INPUT == "Peaks" and self.Config.TYPE == "combined":
            self.Config.NR_OF_GRADIENTS = 3 * self.Config.NR_OF_CLASSES
        else:
            self.Config.NR_OF_GRADIENTS = 33

        if self.Config.LOSS_FUNCTION == "soft_sample_dice":
            self.criterion = pytorch_utils.soft_sample_dice
        elif self.Config.LOSS_FUNCTION == "soft_batch_dice":
            self.criterion = pytorch_utils.soft_batch_dice
        elif self.Config.EXPERIMENT_TYPE == "peak_regression":
            if self.Config.LOSS_FUNCTION == "angle_length_loss":
                self.criterion = pytorch_utils.angle_length_loss
            elif self.Config.LOSS_FUNCTION == "angle_loss":
                self.criterion = pytorch_utils.angle_loss
            elif self.Config.LOSS_FUNCTION == "l2_loss":
                self.criterion = pytorch_utils.l2_loss
        elif self.Config.EXPERIMENT_TYPE == "dm_regression":
            # self.criterion = nn.MSELoss()   # aggregate by mean
            self.criterion = nn.MSELoss(size_average=False, reduce=True)   # aggregate by sum
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        NetworkClass = getattr(importlib.import_module("tractseg.models." + self.Config.MODEL.lower()),
                               self.Config.MODEL)
        
        if self.Config.MODEL == 'UNet_Pytorch' or  self.Config.MODEL == 'UNet_Plus3' or  self.Config.MODEL == 'UNet_Fpn':
            self.net = NetworkClass(img_size = self.Config.img_size, n_input_channels=NR_OF_GRADIENTS, n_classes=self.Config.NR_OF_CLASSES,
                                    n_filt=self.Config.UNET_NR_FILT, batchnorm=self.Config.BATCH_NORM,
                                    dropout=self.Config.USE_DROPOUT, upsample=self.Config.UPSAMPLE_TYPE)
            
        elif self.Config.MODEL =='UNet_SelfCrossAtt':
            config = {}
            config["n_input_channels"] = NR_OF_GRADIENTS
            config["n_classes"] = self.Config.NR_OF_CLASSES
            config["n_filt"]=self.Config.UNET_NR_FILT
            config["hidden_size"]=self.Config.hidden_size
            config["n_tblock"]=self.Config.TransBlock_num
            config["patch_size"]=self.Config.patch_size
            config["n_patch_singledim"]=int(self.Config.INPUT_DIM[0]/self.Config.patch_size)
            config["n_patches"]=int(self.Config.INPUT_DIM[0]/self.Config.patch_size)**2
            config["vis"] = self.Config.Vis_attmap
            config["num_heads"]=self.Config.num_heads
            config["mlp_dim"] = self.Config.mlp_dim
            self.net = NetworkClass(config)

        elif self.Config.MODEL == 'Swin_Unet':
            self.net = NetworkClass(self.Config)
            
        else:
            config = {}
            config["n_input_channels"] = NR_OF_GRADIENTS
            config["n_classes"] = self.Config.NR_OF_CLASSES
            config["n_filt"]=self.Config.UNET_NR_FILT
            config["hidden_size"]=self.Config.hidden_size
            config["n_tblock"]=self.Config.TransBlock_num
            config["patch_size"]=self.Config.patch_size
            config["n_patch_singledim"]=int(self.Config.INPUT_DIM[0]/self.Config.patch_size)
            config["n_patches"]=int(self.Config.INPUT_DIM[0]/self.Config.patch_size)**2
            config["vis"] = self.Config.Vis_attmap
            config["num_heads"]=self.Config.num_heads
            config["mlp_dim"] = self.Config.mlp_dim
            self.net = NetworkClass(config, img_size = self.Config.img_size, n_input_channels=NR_OF_GRADIENTS, n_classes=self.Config.NR_OF_CLASSES,
                    n_filt=self.Config.UNET_NR_FILT, batchnorm=self.Config.BATCH_NORM,
                    dropout=self.Config.USE_DROPOUT, upsample=self.Config.UPSAMPLE_TYPE)


        ##------------------------calculate Params amount and Flops---------------------------
        # input = torch.randn((1,9,144,144))
        # flops, params = profile(self.net, inputs=(input,))
        # print('Type1:--------flops',flops/1e9,'G   ','params',params/1e6,'M--------')
        #

        ##------------ MultiGPU setup---------------------
        nr_gpus = torch.cuda.device_count()
        exp_utils.print_and_save(self.Config.EXP_PATH, "nr of gpus: {}".format(nr_gpus))
        self.net = nn.DataParallel(self.net)



        ##------------------------calculate Params amount and Flops---------------------------
        # num_params = 0
        # for name, param in self.net.named_parameters():
        #     num_params += param.numel()
        # # print('----------The param amount:',num_params/1e6, 'M--------')
        # params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        # model_without_ddp = self.net.module
        # if hasattr(model_without_ddp, 'flops'):
        #     flops = model_without_ddp.flops()
        # print('Type2:--------flops',flops/1e9,'G   ','params',params/1e6,'M--------')
        #
        #

        ###-----------------gpu accelarate------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = self.net.to(self.device)


        if self.Config.OPTIMIZER == "Adamax":
            self.optimizer = Adamax(net.parameters(), lr=self.Config.LEARNING_RATE,
                                    weight_decay=self.Config.WEIGHT_DECAY)
        elif self.Config.OPTIMIZER == "Adam":
            self.optimizer = Adam(net.parameters(), lr=self.Config.LEARNING_RATE,
                                  weight_decay=self.Config.WEIGHT_DECAY)
        else:
            raise ValueError("Optimizer not defined")

        if APEX_AVAILABLE and self.Config.FP16:
            # Use O0 to disable fp16 (might be a little faster on TitanX)
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, verbosity=0, opt_level="O1")
            if not inference:
                print("INFO: Using fp16 training")
        else:
            if not inference:
                print("INFO: Did not find APEX, defaulting to fp32 training")
        
        if self.Config.LR_SCHEDULE:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                            mode=self.Config.LR_SCHEDULE_MODE,
                                                            patience=self.Config.LR_SCHEDULE_PATIENCE)


        # if self.Config.LR_SCHEDULE:
        #     if Config.lr_s == 'default':
        #         self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
        #                                                             mode=self.Config.LR_SCHEDULE_MODE,
        #                                                             patience=self.Config.LR_SCHEDULE_PATIENCE)
        #     if Config.lr_s == 'pla':
        #     ### II: Plateau with warmup
        #         self.scheduler = PlateauLRScheduler(self.optimizer, mode=self.Config.LR_SCHEDULE_MODE,
        #                         patience_t=self.Config.LR_SCHEDULE_PATIENCE, warmup_t=Config.WARMUP_EPOCH,
        #                                             warmup_lr_init=Config.WARMUP_INIT_LR)
        #     if Config.lr_s == 'cosin':
        #         self.scheduler = CosineLRScheduler(self.optimizer, t_initial=Config.NUM_EPOCHS, t_mul=1.,
        #                                            warmup_t=Config.WARMUP_EPOCH, warmup_lr_init=Config.WARMUP_INIT_LR,
        #                                            lr_min=Config.MIN_LR)


        if self.Config.LOAD_WEIGHTS:
            exp_utils.print_verbose(self.Config.VERBOSE, "Loading weights ... ({})".format(self.Config.WEIGHTS_PATH))
            self.load_model(self.Config.WEIGHTS_PATH)

        # Reset weights of last layer for transfer learning
        if self.Config.RESET_LAST_LAYER:
            self.net.conv_5 = nn.Conv2d(self.Config.UNET_NR_FILT, self.Config.NR_OF_CLASSES, kernel_size=1,
                                        stride=1, padding=0, bias=True).to(self.device)


    def train(self, X, y, weight_factor=None):
        X = X.contiguous().cuda(non_blocking=True)  # (bs, features, x, y)
        y = y.contiguous().cuda(non_blocking=True)  # (bs, classes, x, y)

        self.net.train()
        self.optimizer.zero_grad()
        outputs = self.net(X)  # (bs, classes, x, y)
        angle_err = None

        if self.Config.EXPERIMENT_TYPE == "peak_regression":
            f1 = metric_utils.calc_peak_length_dice_pytorch(self.Config.CLASSES, outputs.detach(), y.detach(),
                                                            max_angle_error=self.Config.PEAK_DICE_THR,
                                                            max_length_error=self.Config.PEAK_DICE_LEN_THR)
        elif self.Config.EXPERIMENT_TYPE == "dm_regression":
            f1 = pytorch_utils.f1_score_macro(y.detach() > self.Config.THRESHOLD, outputs.detach(),
                                              per_class=True, threshold=self.Config.THRESHOLD)
        else:
            f1 = pytorch_utils.f1_score_macro(y.detach(), F.sigmoid(outputs).detach(), per_class=True,
                                              threshold=self.Config.THRESHOLD)

        if weight_factor is not None:
            if len(y.shape) == 4:  # 2D
                weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES,
                                      y.shape[2], y.shape[3])).cuda()
            else:  # 3D
                weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES,
                                      y.shape[2], y.shape[3], y.shape[4])).cuda()
            bundle_mask = y > 0
            weights[bundle_mask.data] *= weight_factor  # 10

            if self.Config.EXPERIMENT_TYPE == "peak_regression":
                loss, angle_err = self.criterion(outputs, y, weights)
            else:
                loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, y)
        else:
            if self.Config.LOSS_FUNCTION == "soft_sample_dice" or self.Config.LOSS_FUNCTION == "soft_batch_dice":
                loss = self.criterion(F.sigmoid(outputs), y)
            else:
                loss = self.criterion(outputs, y)




        if APEX_AVAILABLE and self.Config.FP16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()



        if self.Config.USE_VISLOGGER:
            probs = F.sigmoid(outputs)
        else:
            probs = None  # faster

        metrics = {}
        metrics["loss"] = loss.item()
        metrics["f1_macro"] = f1
        metrics["angle_err"] = angle_err if angle_err is not None else 0

        return probs, metrics


    def test(self, X, y, weight_factor=None):
        with torch.no_grad():
            X = X.contiguous().cuda(non_blocking=True)
            y = y.contiguous().cuda(non_blocking=True)

        if self.Config.DROPOUT_SAMPLING:
            self.net.train()
        else:
            self.net.train(False)
        outputs = self.net(X)
        angle_err = None



        if self.Config.EXPERIMENT_TYPE == "peak_regression":
            f1 = metric_utils.calc_peak_length_dice_pytorch(self.Config.CLASSES, outputs.detach(), y.detach(),
                                                            max_angle_error=self.Config.PEAK_DICE_THR,
                                                            max_length_error=self.Config.PEAK_DICE_LEN_THR)
        elif self.Config.EXPERIMENT_TYPE == "dm_regression":
            f1 = pytorch_utils.f1_score_macro(y.detach() > self.Config.THRESHOLD, outputs.detach(),
                                              per_class=True, threshold=self.Config.THRESHOLD)
        else:
            f1 = pytorch_utils.f1_score_macro(y.detach(), F.sigmoid(outputs).detach(), per_class=True,
                                              threshold=self.Config.THRESHOLD)


        if weight_factor is not None:
            if len(y.shape) == 4:  # 2D
                weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES,
                                      y.shape[2], y.shape[3])).cuda()
            else:  # 3D
                weights = torch.ones((self.Config.BATCH_SIZE, self.Config.NR_OF_CLASSES,
                                      y.shape[2], y.shape[3], y.shape[4])).cuda()
            bundle_mask = y > 0
            weights[bundle_mask.data] *= weight_factor
            if self.Config.EXPERIMENT_TYPE == "peak_regression":
                loss, angle_err = self.criterion(outputs, y, weights)
            else:
                loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, y)
        else:
            if self.Config.LOSS_FUNCTION == "soft_sample_dice" or self.Config.LOSS_FUNCTION == "soft_batch_dice":
                loss = self.criterion(F.sigmoid(outputs), y)
                # loss = criterion(F.sigmoid(outputs), y) + nn.BCEWithLogitsLoss()(outputs, y)
            else:
                loss = self.criterion(outputs, y)


        if self.Config.USE_VISLOGGER:
            probs = F.sigmoid(outputs)
        else:
            probs = None  # faster

        metrics = {}
        metrics["loss"] = loss.item()
        metrics["f1_macro"] = f1
        metrics["angle_err"] = angle_err if angle_err is not None else 0

        return probs, metrics


    def predict(self, X):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).contiguous().to(self.device)

        if self.Config.DROPOUT_SAMPLING:
            self.net.train()
        else:
            self.net.train(False)
        outputs = self.net(X)  # forward
        if self.Config.EXPERIMENT_TYPE == "peak_regression" or self.Config.EXPERIMENT_TYPE == "dm_regression":
            probs = outputs.detach().cpu().numpy()
        else:
            probs = F.sigmoid(outputs).detach().cpu().numpy()

        if self.Config.DIM == "2D":
            probs = probs.transpose(0, 2, 3, 1)  # (bs, x, y, classes)
        else:
            probs = probs.transpose(0, 2, 3, 4, 1)  # (bs, x, y, z, classes)
        return probs


    # def save_model(self, metrics, epoch_nr, mode="f1"):
    #     if mode == "f1": ## True
    #         max_f1_idx = np.argmax(metrics["f1_macro_validate"])
    #         max_f1 = np.max(metrics["f1_macro_validate"])
    #         do_save = epoch_nr == max_f1_idx and max_f1 > 0.01
    #
    #     else:
    #         min_loss_idx = np.argmin(metrics["loss_validate"])
    #         # min_loss = np.min(metrics["loss_validate"])
    #         do_save = epoch_nr == min_loss_idx
    #
    #     # saving to network drives takes 5s (to local only 0.5s) -> do not save too often
    #     if do_save:
    #         print("  Saving weights...")
    #         # remove weights from previous epochs
    #         # for fl in glob.glob(join(self.Config.EXP_PATH, "best_weights_ep*")):
    #         #     os.remove(fl)
    #         try:
    #             #Actually is a pkl not a npz
    #
    #             path=join(self.Config.EXP_PATH, "best_weights_ep" + str(epoch_nr) + ".npz")
    #             print('weights saving path is', path)
    #
    #             pytorch_utils.save_checkpoint(join(self.Config.EXP_PATH, "best_weights_ep" + str(epoch_nr) + ".npz"),
    #                                           unet=self.net)
    #         except IOError:
    #             print("\nERROR: Could not save weights because of IO Error\n")
    #         self.Config.BEST_EPOCH = epoch_nr
    def save_model(self, metrics, epoch_nr, mode="f1"):
        if mode == "f1":  ## True
            max_f1_idx = np.argmax(metrics["f1_macro_validate"])
            max_f1 = np.max(metrics["f1_macro_validate"])
            do_save = epoch_nr == max_f1_idx and max_f1 > 0.01

        else:
            min_loss_idx = np.argmin(metrics["loss_validate"])
            # min_loss = np.min(metrics["loss_validate"])
            do_save = epoch_nr == min_loss_idx

        # saving to network drives takes 5s (to local only 0.5s) -> do not save too often
        if do_save:
            print("Saving best weights...")
            # remove weights from previous epochs
            for fl in glob.glob(join(self.Config.EXP_PATH, "best_weights_ep" + str(self.Config.BEST_EPOCH) + ".npz")):
                os.remove(fl)
            try:
                # Actually is a pkl not a npz

                path = join(self.Config.EXP_PATH, "best_weights_ep" + str(epoch_nr) + ".npz")
                print('weights saving path is', path)

                pytorch_utils.save_checkpoint(join(self.Config.EXP_PATH, "best_weights_ep" + str(epoch_nr) + ".npz"),
                                              unet=self.net)
            except IOError:
                print("\nERROR: Could not save weights because of IO Error\n")
            self.Config.BEST_EPOCH = epoch_nr

        if epoch_nr % 50 == 0 and epoch_nr > 100:
            print("Saving 50 weights...")
            path = join(self.Config.EXP_PATH, "best_weights_ep" + str(epoch_nr) + ".npz")
            print('weights saving path is', path)

            pytorch_utils.save_checkpoint(join(self.Config.EXP_PATH, "best_weights_ep" + str(epoch_nr) + ".npz"),
                                          unet=self.net)

    def load_model(self, path):
        if self.Config.RESET_LAST_LAYER:
            pytorch_utils.load_checkpoint_selectively(path, unet=self.net)
        else:
            pytorch_utils.load_checkpoint(path, unet=self.net)

    def print_current_lr(self):
        for param_group in self.optimizer.param_groups:
            exp_utils.print_and_save(self.Config.EXP_PATH, "current learning rate: {}".format(param_group['lr']))

