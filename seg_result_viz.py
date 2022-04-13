# %%
import nibabel as nib
import os
import numpy as np
from matplotlib import pyplot as plt

cwd = "/content/drive/MyDrive/22WN/swin-unet/model_out/predictions/"
if not os.path.exists(cwd+"fig"):
    os.mkdir(cwd+"fig")
data_folder = os.path.join(cwd)
files = os.listdir(os.path.join(data_folder))

Unlabelled = [0, 0, 0]
Spleen = [128, 128, 128]
RightKidney = [128, 0, 0]
LeftKidney = [192, 192, 128]
Gallbladder = [128, 64, 128]
Esophagus = [60, 40, 222]
Liver = [128, 128, 0]
Stomach = [192, 128, 128]
Aorta = [64, 64, 128]
InfVenaCava = [64, 0, 128]
Vein = [64, 64, 0]
Pancreas = [0, 128, 192]
RightAdrenalGland = [128, 40, 60]
LeftAdrenalGland = [0, 40, 60]

COLOR_DICT = np.array([Unlabelled, Spleen, RightKidney, LeftKidney, Gallbladder,
                       Esophagus, Liver, Stomach, Aorta, InfVenaCava, Vein,
                       Pancreas, RightAdrenalGland, LeftAdrenalGland])


def vis(gt, img, pred, case_idx, slice_idx, color_dict):
    gt_slice = gt[slice_idx, :, :]
    img_slice = img[slice_idx, :, :]
    pred_slice = pred[slice_idx, :, :]

    plt.figure(figsize=(12, 5))
    plt.axis('off')
    ax = plt.subplot(131)
    ax.set_axis_off()
    plt.imshow(img_slice, interpolation='nearest')
    plt.title(case_idx+"_"+str(slice_idx))

    ax = plt.subplot(132)
    ax.set_axis_off()
    gt_img = np.zeros(gt_slice.shape + (3,))
    for i in range(len(color_dict)):
        gt_img[gt_slice == i, :] = color_dict[i]
    plt.imshow(gt_img / 255, interpolation='nearest')
    plt.title(case_idx+"_"+str(slice_idx)+"_gt")

    ax = plt.subplot(133)
    ax.set_axis_off()
    pred_img = np.zeros(pred_slice.shape + (3,))
    for i in range(len(color_dict)):
        pred_img[pred_slice == i, :] = color_dict[i]
    plt.imshow(pred_img / 255, interpolation='nearest')
    plt.title(case_idx+"_"+str(slice_idx)+"_pred")
    plt.tight_layout()
    plt.savefig(cwd + "fig/" + case_idx+"_"+str(slice_idx))


gt = nib.load(
    "/content/drive/MyDrive/22WN/transfuse/out/predictions/case0001_gt.nii.gz").get_fdata()
img = nib.load(
    "/content/drive/MyDrive/22WN/transfuse/out/predictions/case0001_img.nii.gz").get_fdata()
img_clipped = np.clip(img, -125, 275)
img_normalised = (img_clipped - (-125)) / (275 - (-125))
pred = nib.load(
    "/content/drive/MyDrive/22WN/transfuse/out/predictions/case0001_pred.nii.gz").get_fdata()

vis(gt, img, pred, "0001", 90, COLOR_DICT)

gt = nib.load(
    "/content/drive/MyDrive/22WN/transfuse/out/predictions/case0008_gt.nii.gz").get_fdata()
img = nib.load(
    "/content/drive/MyDrive/22WN/transfuse/out/predictions/case0008_img.nii.gz").get_fdata()
img_clipped = np.clip(img, -125, 275)
img_normalised = (img_clipped - (-125)) / (275 - (-125))
pred = nib.load(
    "/content/drive/MyDrive/22WN/transfuse/out/predictions/case0008_pred.nii.gz").get_fdata()

vis(gt, img, pred, "0008", 110, COLOR_DICT)
