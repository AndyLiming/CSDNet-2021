import os
import sys
import numpy as np
import torch
import torchvision
import cv2


def saveValImage(leftDepthGt, leftERP, leftDepthPred, e, path):
  b, c, h, w = leftDepthGt.size()
  div = torch.zeros([c, h, 10])
  gt = leftDepthGt[0, ::].cpu()
  erp = leftERP[0, ::].cpu()
  pred = leftDepthPred[0, ::].cpu()
  gt = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
  erp = (erp - torch.min(erp)) / (torch.max(erp) - torch.min(erp))
  pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))
  saveimg = torch.cat([gt, div, erp, div, pred], dim=2)
  prefix = "{:0>3}val2".format(e)
  torchvision.utils.save_image(saveimg, os.path.join(path, prefix + '.png'))


def saveAllTestImage(leftDepthGt, leftDepthPred, invalidMask, coun, path, saveWithGT=False, log=True):
  b, c, h, w = leftDepthGt.size()
  if log:
    div = torch.ones([c, h, 10])
    mask = ~invalidMask
    div = torch.log10(div * 1000 + 1.0)
    leftDepthPred[mask] = torch.log10(leftDepthPred[mask] + 1.0)
    leftDepthGt[mask] = torch.log10(leftDepthGt[mask] + 1.0)
    leftDepthPred[invalidMask] = 0
    leftDepthGt[invalidMask] = 0
    for i in range(b):
      predSave = leftDepthPred[i, ::].cpu()
      gtSave = leftDepthGt[i, ::].cpu()
      maskSave = mask[i, ::].cpu()
      saveimg = torch.cat([gtSave, div, predSave], dim=2).squeeze_(0).numpy()
      saveimg = (saveimg - np.min(saveimg)) / (np.max(saveimg) - np.min(saveimg)) * 255
      saveimg = saveimg.astype(np.uint8)
      prefix = "{:0>4}_test".format(coun + i)
      saveimg = cv2.applyColorMap(saveimg, cv2.COLORMAP_JET)
      cv2.imwrite(os.path.join(path, prefix + '.png'), saveimg)
  else:
    div = torch.zeros([c, h, 10])
    for k in range(b):
      gt = leftDepthGt[k, ::].cpu()
      pred = leftDepthPred[k, ::].cpu()
      mask = invalidMask[k, ::].cpu()
      gt[mask] = 0.0
      pred[mask] = 0.0
      prefix = "{:0>4}testAll-pred".format(coun + k)
      predExr = pred.squeeze_(0).numpy()
      cv2.imwrite(os.path.join(path, prefix + '.exr'), predExr)
      gt = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
      pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))
      saveimg = torch.cat([gt, div, pred], dim=2) if saveWithGT else pred
      torchvision.utils.save_image(saveimg, os.path.join(path, prefix + '.png'))


def saveERPTestImage(leftDepthGt, erp, invalidMask, coun, path):
  b, c, h, w = leftDepthGt.size()
  for k in range(b):
    pred = erp[k, ::].cpu()
    mask = invalidMask[k, ::].cpu()
    pred[mask] = 0.0
    pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))
    saveimg = pred
    prefix = "{:0>4}testERP".format(coun + k)
    torchvision.utils.save_image(saveimg, os.path.join(path, prefix + '.png'))