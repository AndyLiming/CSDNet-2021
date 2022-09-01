import argparse
import os
import sys
import numpy as np
import pickle
import time
import cv2

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from dataset import Dataset360D
from models import CSDNet
import spherical as S360
sys.path.append("../")
from meshcnn.utils import xyz2latlong, interp_r2tos2
epsilon = 1e-10

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', default=16, type=int, help="batch size for training")
parser.add_argument('--dataset_name', default='3D60', type=str, help="dataset name")
parser.add_argument('--filename_list', default=None, type=str, help="file name list for testing")
parser.add_argument('--dataset_root', default=None, type=str, help="file name list for testing")

parser.add_argument('--load_ckpt', default=None, type=str, help="path to the checkpoint file")

parser.add_argument('--spherical_metrics', action='store_true', help="use spherical metrics instead of sphere")
parser.add_argument('--save_output', action='store_true', help="save output depth image or not")
parser.add_argument('--save_dir', default='./outputs/testOutput/', type=str, help="directory to save output images")
parser.add_argument('--width', default=512, type=int, help="width of input and output images")
parser.add_argument('--height', default=256, type=int, help="height of input and output images")

parser.add_argument('--mesh_level', default=6, type=int, help="mesh level of spherical CNNs")
parser.add_argument('--max_depth', default=20, type=float, help="max valid depth")
parser.add_argument('--baseline', default=0.26, type=float, help="baseline of binocular spherical system")
parser.add_argument('--parallel', action='store_true', help="if use data parallel or not")

args = parser.parse_args()

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def compute_errors(gt, pred, invalid_mask, weights, sampling, mode='cpu', median_scale=False):
  b, _, __, ___ = gt.size()
  scale = torch.median(gt.reshape(b, -1), dim=1)[0] / torch.median(pred.reshape(b, -1), dim=1)[0]\
      if median_scale else torch.tensor(1.0).expand(b, 1, 1, 1).to(gt.device)
  pred = pred * scale.reshape(b, 1, 1, 1)
  valid_sum = torch.sum(~invalid_mask, dim=[1, 2, 3], keepdim=True)
  gt[invalid_mask] = 0.0
  pred[invalid_mask] = 0.0
  thresh = torch.max((gt / pred), (pred / gt))
  thresh[invalid_mask | (sampling < 0.5)] = 2.0

  sum_dims = [1, 2, 3]
  delta_valid_sum = torch.sum(~invalid_mask & (sampling > 0), dim=[1, 2, 3], keepdim=True)
  delta1 = (thresh < 1.25).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()
  delta2 = (thresh < (1.25**2)).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()
  delta3 = (thresh < (1.25**3)).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()

  rmse = (gt - pred)**2
  rmse[invalid_mask] = 0.0
  rmse_w = rmse * weights
  rmse_mean = torch.sqrt(rmse_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float())

  rmse_log = (torch.log(gt) - torch.log(pred))**2
  rmse_log[invalid_mask] = 0.0
  rmse_log_w = rmse_log * weights
  rmse_log_mean = torch.sqrt(rmse_log_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float())

  abs_rel = (torch.abs(gt - pred) / gt)
  abs_rel[invalid_mask] = 0.0
  abs_rel_w = abs_rel * weights
  abs_rel_mean = abs_rel_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float()

  sq_rel = (((gt - pred)**2) / gt)
  sq_rel[invalid_mask] = 0.0
  sq_rel_w = sq_rel * weights
  sq_rel_mean = sq_rel_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float()

  return (abs_rel_mean, abs_rel), (sq_rel_mean, sq_rel), (rmse_mean, rmse), (rmse_log_mean, rmse_log), delta1, delta2, delta3


def compute_errors_S2(gt, pred, invalid_mask, mode='cpu', median_scale=False):
  b, _, __ = gt.size()
  scale = torch.median(gt.reshape(b, -1), dim=1)[0] / torch.median(pred.reshape(b, -1), dim=1)[0]\
      if median_scale else torch.tensor(1.0).expand(b, 1, 1).to(gt.device)
  pred = pred * scale.reshape(b, 1, 1)
  valid_sum = torch.sum(~invalid_mask, dim=[1, 2], keepdim=True)
  gt[invalid_mask] = 0.0
  pred[invalid_mask] = 0.0
  thresh = torch.max((gt / pred), (pred / gt))
  thresh[invalid_mask] = 2.0

  sum_dims = [1, 2]
  delta_valid_sum = torch.sum(~invalid_mask, dim=[1, 2], keepdim=True)
  delta1 = (thresh < 1.25).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()
  delta2 = (thresh < (1.25**2)).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()
  delta3 = (thresh < (1.25**3)).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()

  rmse = (gt - pred)**2
  rmse[invalid_mask] = 0.0
  rmse_w = rmse
  rmse_mean = torch.sqrt(rmse_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float())

  rmse_log = (torch.log(gt) - torch.log(pred))**2
  rmse_log[invalid_mask] = 0.0
  rmse_log_w = rmse_log
  rmse_log_mean = torch.sqrt(rmse_log_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float())

  abs_rel = (torch.abs(gt - pred) / gt)
  abs_rel[invalid_mask] = 0.0
  abs_rel_w = abs_rel
  abs_rel_mean = abs_rel_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float()

  sq_rel = (((gt - pred)**2) / gt)
  sq_rel[invalid_mask] = 0.0
  sq_rel_w = sq_rel
  sq_rel_mean = sq_rel_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float()

  return (abs_rel_mean, abs_rel), (sq_rel_mean, sq_rel), (rmse_mean, rmse), (rmse_log_mean, rmse_log), delta1, delta2, delta3


def spiral_sampling(grid, percentage):
  b, c, h, w = grid.size()
  N = torch.tensor(h * w * percentage).int().float()
  sampling = torch.zeros_like(grid)[:, 0, :, :].unsqueeze(1)
  phi_k = torch.tensor(0.0).float()
  for k in torch.arange(N - 1):
    k = k.float() + 1.0
    h_k = -1 + 2 * (k - 1) / (N - 1)
    theta_k = torch.acos(h_k)
    phi_k = phi_k + torch.tensor(3.6).float() / torch.sqrt(N) / torch.sqrt(1 - h_k * h_k) \
        if k > 1.0 else torch.tensor(0.0).float()
    phi_k = torch.fmod(phi_k, 2 * np.pi)
    sampling[:, :, int(theta_k / np.pi * h) - 1, int(phi_k / np.pi / 2 * w) - 1] += 1.0
  return (sampling > 0).float()


def saveOutputOriValue(pred, gt, mask, rootDir, id, names=None, cons=True):
  b, c, h, w = pred.shape
  pred[~mask] = 0
  gt[~mask] = 0
  for i in range(b):
    predSave = pred[i, ::].cpu()
    gtSave = gt[i, ::].cpu()
    maskSave = mask[i, ::].cpu()
    saveimg = predSave.squeeze_(0).numpy()
    if names is None:
      prefix = "{:0>4}_test".format(id + i)
    else:
      oriName = names[i]
      if isinstance(oriName, list) or isinstance(oriName, tuple):
        oriName = oriName[0]
      oriName = oriName.replace(args.dataset_root, '')
      #oriName = oriName.replace(args.intermedia_path, '')
      oriName = oriName.replace('../', '')
      oriName = oriName.replace('./', '')
      oriName = oriName.replace('/', '+')
      prefix = oriName.split('.')[0]
    # cv2.imwrite(os.path.join(rootDir, prefix + '.exr'), saveimg)
    np.save(os.path.join(rootDir, prefix + '.npy'), saveimg)


def saveOutput(pred, gt, mask, rootDir, id, names=None, log=True, cons=True, savewithGt=False):
  b, c, h, w = pred.shape
  div = torch.ones([c, h, 10])
  if log:
    div = torch.log10(div * 1000 + 1.0)
    pred[mask] = torch.log10(pred[mask] + 1.0)
    gt[mask] = torch.log10(gt[mask] + 1.0)
  pred[~mask] = 0
  gt[~mask] = 0
  for i in range(b):
    predSave = pred[i, ::].cpu()
    gtSave = gt[i, ::].cpu()
    maskSave = mask[i, ::].cpu()
    if savewithGt:
      saveimg = torch.cat([gtSave, div, predSave], dim=2).squeeze_(0).numpy()
    else:
      saveimg = predSave.squeeze_(0).numpy()
    saveimg = (saveimg - np.min(saveimg)) / (np.max(saveimg) - np.min(saveimg)) * 255

    saveimg = saveimg.astype(np.uint8)
    if names is None:
      prefix = "{:0>4}_test".format(id + i)
    else:
      oriName = names[i]
      if isinstance(oriName, list) or isinstance(oriName, tuple):
        oriName = oriName[0]
      oriName = oriName.replace(args.dataset_root, '')
      oriName = oriName.replace('../', '')
      oriName = oriName.replace('./', '')
      oriName = oriName.replace('/', '+')
      prefix = oriName.split('.')[0]
    saveimg = cv2.applyColorMap(saveimg, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(rootDir, prefix + '.png'), saveimg)


def main():
  testFileList = args.filename_list
  saveDir = args.save_dir
  os.makedirs(saveDir, exist_ok=True)
  maxDepth = args.max_depth
  batchSize = args.batch_size
  meshLevel = args.mesh_level
  width, height = args.width, args.height
  print("use sphere metrics: {}".format(args.spherical_metrics))
  print("save output images: {}".format(args.save_output))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  testMode = 'gpu' if torch.cuda.is_available() else 'cpu'
  if args.dataset_name == '3D60':
    testData = Dataset360D(filenamesFile=testFileList, delimiter=" ", mode="lr", inputShape=[height, width], meshLevel=meshLevel)
  else:
    raise NotImplementedError("dataset {} is not support yet!".format(args.dataset_name))
  testDataLoader = torch.utils.data.DataLoader(testData, batch_size=batchSize, num_workers=8, pin_memory=False, shuffle=False)
  error_names = ['abs_rel', 'sq_rel', 'rmse', 'log_rmse', 'delta1', 'delta2', 'delta3']
  num_test_samples = len(testData)

  # model hyper-parameters
  sphereDown = 3
  sphereRes = 3
  sphereFdim = 16
  refineDown = 1
  refineRes = 2
  refineFdim = 16
  # model = torch.load(args.load_ckpt)
  # torch.save(model.state_dict(), 'csdnet_mat3d.pth')
  model = CSDNet(in_ch=2 * 3,
                 out_ch=1,
                 max_level=meshLevel,
                 min_level=meshLevel - sphereDown,
                 fdim=sphereFdim,
                 resNum=sphereRes,
                 refineDownNum=refineDown,
                 refineResNum=refineRes,
                 refineFdim=refineFdim,
                 dense=True,
                 parallel=args.parallel)
  state_dict = torch.load(args.load_ckpt)
  f = lambda x: x.split('module.', 1)[-1] if x.startswith('module') else x
  state_dict = {f(key): value for key, value in state_dict.items()}
  # model.load_state_dict(load_dict)
  model.load_state_dict(state_dict)
  # parallel
  if args.parallel:
    model = nn.DataParallel(model)
  model.to(device)
  print("test dataset: {}".format(testFileList))
  print("num of test images: {}".format(num_test_samples))
  errorsS2 = np.zeros((len(error_names), num_test_samples), np.float32)
  errorsERP = np.zeros((len(error_names), num_test_samples), np.float32)
  errorsPred = np.zeros((len(error_names), num_test_samples), np.float32)
  errorsS2Final = np.zeros((len(error_names), num_test_samples), np.float32)

  if args.spherical_metrics:
    # weights = S360.weights.spherical_confidence(S360.grid.create_spherical_grid(width)).to(device)
    weights = S360.weights.theta_confidence(S360.grid.create_spherical_grid(width)).to(device)
    sampling = spiral_sampling(S360.grid.create_image_grid(width, height), 0.25).to(device)
  else:
    weights = torch.ones(1, 1, height, width).to(device)
    sampling = torch.ones(1, 1, height, width).to(device)

  meshFile = pickle.load(open('./meshfiles/icosphere_{}.pkl'.format(meshLevel), 'rb'))
  V = meshFile['V']
  model.eval()
  counter = 0
  num = 0
  startTime = time.time()
  with torch.no_grad():
    for numB, batchData in enumerate(tqdm(testDataLoader, desc="CSDNet Testing")):
      leftRgb = batchData['leftRGB'].to(device)
      rightRgb = batchData['rightRGB'].to(device)
      leftDepthGt = batchData['leftDepth'].to(device)
      leftRgbS2 = batchData['leftRGBS2']
      rightRgbS2 = batchData['rightRGBS2']
      leftDepthGtS2 = batchData['leftDepthS2'].to(device)

      b, c, h, w = leftRgb.size()

      input = torch.cat([leftRgbS2, rightRgbS2], dim=1).to(device)
      inputImg = torch.cat([leftRgb, rightRgb], dim=1).to(device)

      invalidMask = ((leftDepthGt > maxDepth) | (leftDepthGt <= 0) | torch.isnan(leftDepthGt))
      #
      invalidMaskS2 = ((leftDepthGtS2 > maxDepth) | (leftDepthGtS2 <= 0) | torch.isnan(leftDepthGtS2))

      predS2, erp, predDep = model(input, inputImg)

      predDep[predDep < epsilon] = epsilon
      erp[erp < epsilon] = epsilon
      predS2[predS2 < epsilon] = epsilon
      tmp = []
      for j in range(b):
        tt = torch.from_numpy(interp_r2tos2(predDep[j, ::].squeeze(0).cpu(), V)).to(device)
        tt.unsqueeze_(0)
        tt.unsqueeze_(0)
        tmp.append(tt)
      predDepS2 = torch.cat([x for x in tmp], dim=0)

      abs_rel_t, sq_rel_t, rmse_t, rmse_log_t, delta1, delta2, delta3 = compute_errors(leftDepthGt, predDep, invalidMask, weights=weights, sampling=sampling, mode=testMode)
      for i in range(b):
        idx = counter + i
        errorsPred[:, idx] = abs_rel_t[0][i].cpu(), sq_rel_t[0][i].cpu(), rmse_t[0][i].cpu(), rmse_log_t[0][i].cpu(), delta1[i].cpu(), delta2[i].cpu(), delta3[i].cpu()

      abs_rel_t, sq_rel_t, rmse_t, rmse_log_t, delta1, delta2, delta3 = compute_errors_S2(leftDepthGtS2, predS2, invalidMaskS2, mode=testMode)
      for i in range(b):
        idx = counter + i
        errorsS2[:, idx] = abs_rel_t[0][i].cpu(), sq_rel_t[0][i].cpu(), rmse_t[0][i].cpu(), rmse_log_t[0][i].cpu(), delta1[i].cpu(), delta2[i].cpu(), delta3[i].cpu()

      abs_rel_t, sq_rel_t, rmse_t, rmse_log_t, delta1, delta2, delta3 = compute_errors_S2(leftDepthGtS2, predDepS2, invalidMaskS2, mode=testMode)
      for i in range(b):
        idx = counter + i
        errorsS2Final[:, idx] = abs_rel_t[0][i].cpu(), sq_rel_t[0][i].cpu(), rmse_t[0][i].cpu(), rmse_log_t[0][i].cpu(), delta1[i].cpu(), delta2[i].cpu(), delta3[i].cpu()

      abs_rel_t, sq_rel_t, rmse_t, rmse_log_t, delta1, delta2, delta3 = compute_errors(leftDepthGt, erp, invalidMask, weights=weights, sampling=sampling, mode=testMode)
      for i in range(b):
        idx = counter + i
        errorsERP[:, idx] = abs_rel_t[0][i].cpu(), sq_rel_t[0][i].cpu(), rmse_t[0][i].cpu(), rmse_log_t[0][i].cpu(), delta1[i].cpu(), delta2[i].cpu(), delta3[i].cpu()

      if args.save_output:
        saveOutputOriValue(predDep, leftDepthGt, ~invalidMask, saveDir, counter, names=batchData['leftNames'])
        saveOutput(predDep, leftDepthGt, ~invalidMask, saveDir, counter, names=batchData['leftNames'])
      num += 1
      counter += b
    mean_errorsPred = errorsPred.mean(1)
    mean_errorsERP = errorsERP.mean(1)
    mean_errorsS2 = errorsS2.mean(1)
    mean_errorS2_final = errorsS2Final.mean(1)

    print("Results (test on: {} - {}, mesh {} with Refine): ".format('CSDNet', args.load_ckpt, meshLevel))
    print("\t{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("\t{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errorsPred))

    print("Results (test on: {} - {}, mesh {} s2 signal): ".format('CSDNet', args.load_ckpt, meshLevel))
    print("\t{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("\t{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errorsS2))

    print("Results (test on: {} - {}, mesh {} s2 signal final): ".format('CSDNet', args.load_ckpt, meshLevel))
    print("\t{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("\t{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errorS2_final))

    print("Results (test on: {} - {}, mesh {} without Refine): ".format('CSDNet', args.load_ckpt, meshLevel))
    print("\t{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("\t{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errorsERP))

    endTime = time.time()
    print("testing time: {}".format(endTime - startTime))


if __name__ == "__main__":
  main()
