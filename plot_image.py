# %%
import nibabel as nib
import os
import numpy as np
from matplotlib import pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)
cwd = "/home/marco/Documents/TransFuse/out/predictions/"
if not os.path.exists(cwd+"fig"):
    os.mkdir(cwd+"fig")
data_folder = os.path.join(cwd)
plt.xticks(fontsize=14)
files = os.listdir(os.path.join(data_folder))
files = ["case0008_img.nii.gz", "case0008_gt.nii.gz", "case0008_pred.nii.gz", "case0001_img.nii.gz", "case0001_gt.nii.gz", "case0001_pred.nii.gz"]
i = 0
while i < len(files):

    gt = nib.load(os.path.join(data_folder, files[i])).get_fdata()
    gt_clipped = np.clip(gt, -125, 275)
    gt_normalised = (gt_clipped - (-125)) / (275 - (-125))
    img = nib.load(os.path.join(data_folder, files[i+1])).get_fdata()
    img_clipped = np.clip(img, -125, 275)
    img_normalised = (img_clipped - (-125)) / (275 - (-125))
    pred = nib.load(os.path.join(data_folder, files[i+2])).get_fdata()
    pred_clipped = np.clip(pred, -125, 275)
    pred_normalised = (pred_clipped - (-125)) / (275 - (-125))
    casename = files[i][:8]
    for j in range(gt_normalised.shape[2]): 
        if j == 110 or j == 90:
            plt.figure(1)
            plt.subplot(131)
            print(gt.shape)
            plt.imshow(gt[j,:,:,:], interpolation='nearest')
            plt.subplot(132)
            plt.imshow(img[j,:,:,:], interpolation='nearest')
            plt.subplot(133)   
            pred[j,0,0] = 13
                   
            plt.imshow(pred[j,:,:,:], interpolation='nearest')
            plt.title(casename+"_"+str(j))
            plt.tight_layout()
            plt.savefig(cwd + "fig/" + casename+"_"+str(j), dpi=500)
            # print(gt[300, 150, j])
            # print(pred[300, 150, j])
            print(np.max(pred[j,:,:]))
            print(np.max(gt[j,:,:]))
    i += 3


# %%



