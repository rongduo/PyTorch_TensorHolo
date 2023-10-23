import os
from utils_yujie import ensure_dirs
import argparse
import json
import shutil

def get_config(phase):
    config = Config(phase)
    return config


class Config(object):
    """Base class of Config, provide necessary hyperparameters. 
    """
    def __init__(self, phase):
        self.is_train = phase == "train"

        # init hyperparameters and parse from command-line
        parser, args = self.parse()

        # set as attributes
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
        self.chan_strs = ('red', 'green', 'blue', 'rgb')
        self.wavelength_list = [450*nm, 520*nm, 638*nm]
        self.feature_size = (8*um, 8*um)

        ''' add compression config '''
        self.lambda_schedule = dict(vals=[1, 0.5], steps=[10000]) 
        self.lambda_A = 2**(-4)
        self.lambda_B = 2**(-5)
        self.target_rate = 2.
        self.target_schedule = dict(vals=[0.20/0.14, 1.], steps=[10000])  # Rate allowance

        # experiment paths
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        if args.model_name == '':
            self.log_dir = os.path.join(self.exp_dir, 'log')
            self.model_dir = os.path.join(self.exp_dir, 'model')
        else:
            self.log_dir = os.path.join(self.exp_dir, 'log_{}'.format(args.model_name))
            self.model_dir = os.path.join(self.exp_dir, 'model_{}'.format(args.model_name))

        if args.stage2:
            self.log_dir = self.log_dir +'_stage2'
            self.model_dir = self.model_dir + '_stage2'

        print("----Experiment Configuration-----")
        for k, v in self.__dict__.items():
            print("{0:20}".format(k), v)
            
        if phase == "train" and args.cont is not True and os.path.exists(self.log_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.log_dir)
            shutil.rmtree(self.model_dir)
        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        # create soft link to experiment log directory
        if not os.path.exists('train_log'):
            os.symlink(self.exp_dir, 'train_log')
        
            

        # save this configuration
        if self.is_train:
            if args.model_name != '':
                log_file = 'train_log/config_{}.txt'.format(args.model_name)
            else:
                log_file = 'train_log/config.txt'
            with open(log_file, 'w') as f:
                json.dump(self.__dict__, f, indent=2)


    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        
        # basic configuration
        self._add_basic_config_(parser)

        # dataset configuration
        self._add_dataset_config_(parser)

        # reconstruction cofiguration
        self.__add_recon_config_(parser)

        # model configuration
        self._add_network_config_(parser)

        # training or testing configuration
        self._add_training_config_(parser)
        
        self._add_test_config_(parser)

        # additional parameters if needed
        pass

        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        """add general hyperparameters"""
        group = parser.add_argument_group('basic')
        group.add_argument('--proj_dir', type=str, default="/mnt/data/home/yujie/yujie_codes/TENSOR_HOLO", help="path to project folder where models and logs will be saved")
        group.add_argument('--holo_data_root', type=str, default='/mnt/data/home/yujie/yujie_data/MIT_CGH_4K')
        group.add_argument('--exp_name', type=str, default=os.getcwd().split('/')[-1], help="name of this experiment")
        group.add_argument('-g', '--gpu_ids', type=str, default='4', help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

    def _add_dataset_config_(self, parser):
        """add hyperparameters for dataset configuration"""       
        group = parser.add_argument_group('dataset')
        group.add_argument('--batch_size', type=int, default=1, help="batch size")
        group.add_argument('--num_workers', type=int, default=4, help="number of workers for data loading")
        group.add_argument('--channel', type=str, choices=['r','g','b', 'rgb'], help="select which channel to train")
        group.add_argument('--img_res', type=int, default=384, choices=[192, 384], help='which resolution of the data to load')
        
    def _add_network_config_(self, parser):
        """add hyperparameters for network architecture"""
        group = parser.add_argument_group('network')
        group.add_argument('--pretrain_path', type=str, default='')
        group.add_argument('--stage2', type=bool, default=False)
        group.add_argument('--n_layers', type=int, default=16, help="the amount of the layers within the residual network")
        group.add_argument('--input_dim', type=int, default=4, help='input dimension of the network')
        group.add_argument('--output_dim', type=int, default=6, help='output dimension of the network')
        group.add_argument('--inter_dim', type=int, default=24, help='number of channels for the intermediate layers')
        group.add_argument('--kernel_size', type=int, default=3, help='kernel size of the network covolutional layers')
        group.add_argument('--norm_type', type=str, default='batch', choices=['batch', 'instance'], help='normalization type of the network')

    def _add_training_config_(self, parser):
        """training configuration"""
        group = parser.add_argument_group('training')
        group.add_argument('--nr_epochs', type=int, default=1000, help="total number of epochs to train")
        group.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")
        group.add_argument('--lr_D', type=float, default=1e-4, help="initial learning rate")
        group.add_argument('--lr_step_size', type=int, default=5, help="step size for learning rate decay")
        group.add_argument('--continue', dest='cont',  action='store_true', help="continue training from checkpoint")
        group.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        group.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
        group.add_argument('--save_frequency', type=int, default=30, help="save models every x epochs")
        group.add_argument('--val_frequency', type=int, default=5, help="run validation every x iterations")
        group.add_argument('--vis_frequency', type=int, default=100, help="visualize output every x iterations")
        group.add_argument('--model_name', type=str, default='', help='specify a mdoel name for save the model and log')
        group.add_argument('--compress', action='store_true')
    
    def __add_recon_config_(self, parser):
        """reconstruction configuration"""
        group = parser.add_argument_group('reconstruction')
        group.add_argument('--n_fixed_depth', type=int, default=15, help='number of fixed distances for reconstruction')
        group.add_argument('--n_float_depth', type=int, default=5, help='number of float distances for reconstruction')
        group.add_argument('--depth_scale', type=float, default=6, help='the width of the volume between near and far plane')
        group.add_argument('--depth_base', type=float, default=-3, help='the distance value for near cliping plane')

        #group.add_argument('--pretrain_path', type=str, default='/jixie/yujie_codes/HOLOGRAM/img_invertibleGrayscale_v3/model/latest.pth', help='location of the pretrained model.')
        
    def _add_test_config_(self, parser):
        """testing configuration"""
        group = parser.add_argument_group('testing')
        #group.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        group.add_argument('-o', '--output', type=str, default='output folder to save results')
        group.add_argument('--postfix', type=str, default='', help='postfix for name the output')
        group.add_argument('--add_noise', type=bool, default=False, help='whether add noise to latent map or not')
        group.add_argument('--latent_quality', type=int, default=98, help='quality parameter to save latent map in jpeg format')

