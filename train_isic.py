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

# def structure_loss(pred, mask):
#     mask = mask.type('torch.cuda.FloatTensor')
#     mask = mask.reshape(mask.shape[0], -1, mask.shape[1], mask.shape[2])
#     weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
#     wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask)*weit).sum(dim=(2, 3))
#     union = ((pred + mask)*weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1)/(union - inter+1)
#     return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, best_loss, opt, iter_num):
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(opt.num_class)
    for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce_4 = ce_loss(outputs[0], label_batch[:].long())
            loss_ce_3 = ce_loss(outputs[1], label_batch[:].long())
            loss_ce_2 = ce_loss(outputs[2], label_batch[:].long())
            loss_ce = 0.5 * loss_ce_2 + 0.3 * loss_ce_3 + 0.2 * loss_ce_4
            loss_dice_4 = dice_loss(outputs[0], label_batch, softmax=True)
            loss_dice_3 = dice_loss(outputs[1], label_batch, softmax=True)
            loss_dice_2 = dice_loss(outputs[2], label_batch, softmax=True)
            loss_dice = 0.5 * loss_dice_2 + 0.3 * loss_dice_3 + 0.2 * loss_dice_4
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
            optimizer.step()

            iter_num = iter_num + 1
            print('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))



# def test(model, path):

#     model.eval()
#     mean_loss = []

#     for s in ['val', 'test']:
#         image_root = '{}/data_{}.npy'.format(path, s)
#         gt_root = '{}/mask_{}.npy'.format(path, s)
#         test_loader = test_dataset(image_root, gt_root)

#         dice_bank = []
#         iou_bank = []
#         loss_bank = []
#         acc_bank = []

#         for i in range(test_loader.size):
#             image, gt = test_loader.load_data()
#             image = image.cuda()

#             with torch.no_grad():
#                 _, _, res = model(image)
#             loss = structure_loss(res, torch.tensor(gt).unsqueeze(0).unsqueeze(0).cuda())

#             res = res.sigmoid().data.cpu().numpy().squeeze()
#             gt = 1*(gt>0.5)            
#             res = 1*(res > 0.5)

#             dice = mean_dice_np(gt, res)
#             iou = mean_iou_np(gt, res)
#             acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

#             loss_bank.append(loss.item())
#             dice_bank.append(dice)
#             iou_bank.append(iou)
#             acc_bank.append(acc)
            
#         print('{} Loss: {:.4f}, Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
#             format(s, np.mean(loss_bank), np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))

#         mean_loss.append(np.mean(loss_bank))

#     return mean_loss[0] 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=150, help='epoch number')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
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
    parser.add_argument('-o', '--log-path', type=str, default= '/home/marco/Documents/TransFuse/log/', help='log path')
    parser.add_argument('--resume', action="store_true", help='resume training')
    parser.add_argument('--checkpoint_model', type=str, default='Transfuse', help='model_name for test')
    opt = parser.parse_args() 

    # ---- build models ----
    model = TransFuse_S(pretrained=True, num_classes=opt.num_class).cuda()
    if opt.resume:
        msg = model.load_state_dict(torch.load(opt.log_path + opt.checkpoint_model))
        print("Successfully Loaded Checkpoint")
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))
     
    # image_root = '{}/data_train.npy'.format(opt.train_path)
    # gt_root = '{}/mask_train.npy'.format(opt.train_path)

    # train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize)
    db_train = Synapse_dataset(base_dir=opt.data_path, list_dir=opt.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[224, 224])]))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    train_loader = DataLoader(db_train, batch_size=opt.batchsize, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    best_loss = 1e5
    for epoch in range(1, opt.epoch + 1):
        best_loss = train(train_loader, model, optimizer, epoch, best_loss, opt, epoch)
        print(epoch)
        save_interval = 10
        if epoch == 1:
            save_mode_path = os.path.join(opt.log_path + 'Transfuse_epoch_' + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_mode_path)

        if (epoch + 1) % save_interval == 0:
            save_mode_path = os.path.join(opt.log_path + 'Transfuse_epoch_' + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_mode_path)

        if epoch >= opt.epoch - 1:
            save_mode_path = os.path.join(opt.log_path + 'Transfuse_epoch_' + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            break