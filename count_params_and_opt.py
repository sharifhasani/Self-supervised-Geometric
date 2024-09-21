from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import warnings
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from vehiclereid.data_manager import ImageDataManager
from vehiclereid import models
from vehiclereid.losses import CrossEntropyLoss, TripletLoss, EquivarianceConstraintLoss, DeepSupervision, OFPenalty
from vehiclereid.utils.iotools import check_isfile
from vehiclereid.utils.avgmeter import AverageMeter
from vehiclereid.utils.loggers import Logger, RankLogger
from vehiclereid.utils.torchtools import count_num_param, accuracy, \
    load_pretrained_weights, save_checkpoint, resume_from_checkpoint
from vehiclereid.utils.visualtools import visualize_ranked_results
from vehiclereid.utils.rotation_utils import randomly_rotate_images, randomly_rotate_images_3
from vehiclereid.utils.generaltools import set_random_seed
from vehiclereid.eval_metrics import evaluate
from vehiclereid.optimizers import init_optimizer
from vehiclereid.lr_schedulers import init_lr_scheduler
from synbn.sync_batchnorm import convert_model
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from torch import amp
    from apex.multi_tensor_apply import multi_tensor_applier
    use_apex = True
except ImportError:
    # raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
    use_apex = False

# global variables
parser = argument_parser()
args = parser.parse_args()


def main():
    global args
    global use_apex
    if not args.use_apex:
        use_apex = False
    # torch.autograd.set_detect_anomaly(True)
    print(args)
    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    if args.evaluate and args.load_weights:
        save_dir = os.path.dirname("./model-199.pth")
    else:
        save_dir = './logs/' + str(datetime.datetime.now())[:19].replace(':', '-')
    if args.evaluate:
        if args.target_names[0] == 'vehicleID':
            log_name = 'log_test_{}.txt'.format(args.test_size)
        else:
            log_name = 'log_test.txt'
    else:
        log_name = 'log_train.txt'
    # sys.stdout = Logger(osp.join(save_dir, log_name))
    print('==========\nArgs:{}\n=========='.format(args))

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    print('Initializing model: {}'.format(args.arch))
    model = models.init_model(name=args.arch, num_classes=1100, loss={'xent', 'htri'},
                              pretrained=not args.no_pretrained, use_gpu=use_gpu, args=args)
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input

    # Counting Parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Counting FLOPs
    from thop import profile
    flops, _ = profile(model, inputs=(input_tensor,))

    print(f"Number of parameters: {num_params}")
    print(f"Number of FLOPs: {flops}")



if __name__ == '__main__':
    main()
