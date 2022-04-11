import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
from lib.TransFuse import TransFuse_S
# from utils.dataloader import get_loader, test_dataset
from utils.utils import AvgMeter
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from test_isic import mean_dice_np, mean_iou_np
import os
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.utils import DiceLoss
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
from utils.utils import test_single_volume, test_single_volume_rev

def inference(args, model, test_save_path=None):
    db_test = None
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol",list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    print("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        # print(image.shape)
        metric_i = None
        metric_i = test_single_volume(image, label, model, classes=args.num_class, patch_size=[192, 256],
                                      test_save_path=test_save_path, case=case_name, z_spacing=1)

        metric_list += np.array(metric_i)
        print('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_class):
        print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=150, help='epoch number')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_path', type=str,
                        default='data/', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='data/', help='path to test dataset')
    parser.add_argument('--train_save', type=str, default='TransFuse_S')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
    parser.add_argument('--data_path', type=str, default= '/home/marco/Documents/TransFuse/datasets/Synapse/train_npz/', help='dataset path')
    parser.add_argument('--list_dir', type=str, default= '/home/marco/Documents/TransFuse/lists/lists_Synapse', help='list_dir')
    parser.add_argument('--num_class', type=int, default=14, help='number of segmentation classes')
    parser.add_argument('-o', '--log_path', type=str, default= '/home/marco/Documents/TransFuse/log/', help='log path')
    parser.add_argument('--volume_path', type=str,
                    default='/home/marco/Documents/TransFuse/datasets/Synapse/test_vol_h5/', help='dir for validation volume data')
    parser.add_argument('--test_model', type=str, default='Transfuse', help='model_name for test')
    parser.add_argument('--output_dir', type=str, default='/home/marco/Documents/TransFuse/out/', help='output dir')  
    parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
    opt = parser.parse_args() 


    # ---- build models ----
    model = TransFuse_S(pretrained=True, num_classes=opt.num_class).cuda()
    msg = model.load_state_dict(torch.load(opt.log_path + opt.test_model))


    if opt.is_savenii:
        opt.test_save_dir = os.path.join(opt.output_dir, "predictions")
        test_save_path = opt.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(opt, model, test_save_path)