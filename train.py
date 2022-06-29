#! /usr/bin/python3

import argparse
import os
import sys
import numpy as np
import pickle
import gc

gc.enable()

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import time

from dataset import Dataset360D
from utils import loss, io, projection
from models import CSDNet, modelParaInit
import spherical as S360
from torchsummary import summary
sys.path.append("../")
from meshcnn.utils import xyz2latlong, interp_r2tos2

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def main():
  # parameters
  parser = argparse.ArgumentParser()
  parser.add_argument('-b', '--batch_size', default=8, type=int, help="batch size for training")
  parser.add_argument('--dataset_name', default='3D60', type=str, help="dataset name")
  parser.add_argument('--train_files', default='./dataset/train.txt', type=str, help="file name list for training")
  parser.add_argument('--val_files', default='./dataset/test.txt', type=str, help="file name list for evaluation")
  parser.add_argument('-lr', '--learning_rate', default=0.008, type=float, help="initial learning rate for training")
  parser.add_argument('--max_epoch', default=100, type=int, help="max epochs for training")
  parser.add_argument('--width', default=512, type=int, help="width of input and output images")
  parser.add_argument('--height', default=256, type=int, help="height of input and output images")
  parser.add_argument('--channel', default=3, type=int, help="channels of input image, usually set 3 for RGB image and 1 for gray")
  parser.add_argument('--mesh_level', default=6, type=int, help="mesh level of spherical CNNs")
  parser.add_argument('--weight_s2', default=2, type=float, help="weight of sperical loss")
  parser.add_argument('--weight_img', default=5, type=float, help="weight of depth map loss")
  parser.add_argument('--weight_smooth', default=1, type=float, help="weight of smooth loss")
  parser.add_argument('--max_depth', default=20, type=float, help="max valid depth")
  parser.add_argument('--baseline', default=0.26, type=float, help="baseline of binocular spherical system")
  parser.add_argument('--parallel', action='store_true', help="if use data parallel or not")
  parser.add_argument('--save_prefix', default='csdnet', type=str)
  parser.add_argument('--resume_name', default=None, type=str, help="")

  args = parser.parse_args()

  # fix random seed
  torch.manual_seed(100)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(100)

  # parse arguments
  width, height = args.width, args.height
  channel = args.channel
  baseline = args.baseline
  trainFileList = args.train_files
  valFileList = args.val_files
  maxEpoch = args.max_epoch
  meshLevel = args.mesh_level
  batchSize = args.batch_size
  maxDepth = args.max_depth
  logDir = './logs/'
  saveImgDir = './outputs/'
  saveModelDir = './outputs/model'
  os.makedirs(logDir, exist_ok=True)
  os.makedirs(saveImgDir, exist_ok=True)
  os.makedirs(saveModelDir, exist_ok=True)

  meshVerticesMap = {0: 12, 1: 42, 2: 162, 3: 642, 4: 2562, 5: 10242, 6: 40962, 7: 163842}

  sphereDown = 3
  sphereRes = 3
  sphereFdim = 16
  refineDown = 1
  refineRes = 2
  refineFdim = 16
  saveModelName = '{}-mesh_l{}'.format(args.save_prefix, meshLevel)

  # device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  useSphereLossWeight = True

  if useSphereLossWeight:
    weights = S360.weights.theta_confidence(S360.grid.create_spherical_grid(width)).to(device)
  else:
    weights = torch.ones(1, 1, height, width).to(device)

  print("max epochs: {}, initial learning rate: {}".format(args.max_epoch, args.learning_rate))
  print("mesh level: {}".format(args.mesh_level))
  print("weights s2: {}, img: {}, smooth: {}".format(args.weight_s2, args.weight_img, args.weight_smooth))
  print("parallel: {}".format(args.parallel))
  # prepare data
  if args.dataset_name == '3D60':
    trainData = Dataset360D(filenamesFile=trainFileList, delimiter=" ", mode="lr", inputShape=[height, width], meshLevel=meshLevel)
    valData = Dataset360D(valFileList, delimiter=" ", mode="lr", inputShape=[height, width], meshLevel=meshLevel)
  else:
    raise NotImplementedError("dataset {} is not support yet!".format(args.dataset_name))
  trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size=batchSize, num_workers=8, pin_memory=False, shuffle=True)
  valDataLoader = torch.utils.data.DataLoader(valData, batch_size=batchSize, num_workers=8, pin_memory=False, shuffle=False)
  print("dataset: {}. dataloader finished".format(args.dataset_name))
  # prepare model, optimizer, scheduler
  model = CSDNet(in_ch=2 * channel,
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
  # init weights
  # types: 'xavier', 'kaiming','gaussian'
  modelParaInit(model, type='xavier')
  startEpoch = 0
  if args.resume_name is not None:
    model = torch.load(args.resume_name)
  # parallel
  if args.parallel:
    model = nn.DataParallel(model)

  optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
  # optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate)
  model.to(device)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.5)

  # summary & log
  writer = SummaryWriter(log_dir=logDir)
  # try:
  #   summary(model, [(2 * channel, meshVerticesMap[meshLevel]), (2 * channel, height, width)])
  # except Exception as e:
  #   print(e)
  meshFile = pickle.load(open('./meshfiles/icosphere_{}.pkl'.format(meshLevel), 'rb'))
  V = meshFile['V']
  ele, azi = xyz2latlong(V)
  u = ((ele + np.pi / 2) / np.pi * height).astype(np.float32)
  v = ((azi + np.pi) / np.pi * height).astype(np.float32)
  u_ = (u - np.min(u)) / (np.max(u) - np.min(u)) * 2 - 1
  v_ = (v - np.min(u)) / (np.max(v) - np.min(v)) * 2 - 1
  u_ = torch.from_numpy(u_).unsqueeze_(1).unsqueeze_(1)
  v_ = torch.from_numpy(v_).unsqueeze_(1).unsqueeze_(1)
  basegrid = torch.cat([v_, u_], 2)

  sgrid = S360.grid.create_spherical_grid(width).to(device)
  uvgrid = S360.grid.create_image_grid(width, height).to(device)

  for e in range(startEpoch, maxEpoch):
    print("start epoch {}".format(e))
    print("current learning rate: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
    totalLoss = 0.0
    totalS2Loss = 0.0
    count = 0
    # train
    model.train()
    startTime = time.time()
    for i, batchData in enumerate(trainDataLoader):
      optimizer.zero_grad()

      leftRgbS2 = batchData['leftRGBS2']
      rightRgbS2 = batchData['rightRGBS2']
      leftRgb = batchData['leftRGB'].to(device)
      rightRgb = batchData['rightRGB'].to(device)
      leftDepthGt = batchData['leftDepth'].to(device)
      leftDepthGtS2 = batchData['leftDepthS2'].to(device)

      b, c, h, w = leftRgb.size()
      grid = basegrid.repeat(b, 1, 1, 1).to(device)

      input = torch.cat([leftRgbS2, rightRgbS2], dim=1).to(device)
      inputImg = torch.cat([leftRgb, rightRgb], dim=1).to(device)

      outS2, outErp, outImg = model(input, inputImg)

      predS2 = torch.nn.functional.grid_sample(outImg, grid, align_corners=True).squeeze_(3)

      invalidMask = ((leftDepthGt > maxDepth) | (leftDepthGt <= 0) | torch.isnan(leftDepthGt))
      invalidMaskS2 = ((leftDepthGtS2 > maxDepth) | (leftDepthGtS2 <= 0) | torch.isnan(leftDepthGtS2))

      mask = ~invalidMask
      maskS2 = ~invalidMaskS2

      # for smooth loss
      left_xyz = S360.cartesian.coords_3d(sgrid, outImg)
      dI_dxyz = S360.derivatives.dV_dxyz(left_xyz)
      guidance_duv = S360.derivatives.dI_duv(leftRgb).to(device)

      # loss
      # s2Loss = loss.depLoss(outS2, leftDepthGtS2, maxDepth)
      s2Loss = loss.berhuLossS2(outS2, leftDepthGtS2, maskS2) + loss.berhuLossS2(predS2, leftDepthGtS2, maskS2)
      imgLoss = loss.berhuLoss(outImg, leftDepthGt, mask, weights) + loss.berhuLoss(outErp, leftDepthGt, mask, weights)
      smoothLoss = loss.guided_smoothness_loss(dI_dxyz, guidance_duv, mask)

      tLoss = args.weight_s2 * s2Loss + args.weight_img * imgLoss + args.weight_smooth * smoothLoss
      totalLoss += tLoss.data.item()
      totalS2Loss += s2Loss.data.item()

      # backward and step
      tLoss.backward()
      optimizer.step()

      count += 1
    totalLoss /= count
    totalS2Loss /= count
    trainEndTime = time.time()
    print("epoch: {}, train time cost: {} s, average train depth loss:{}, img loss:{}, s2 loss:{}, smooth loss:{}".format(e,
                                                                                                                          trainEndTime - startTime,
                                                                                                                          totalLoss,
                                                                                                                          imgLoss.data.item(),
                                                                                                                          s2Loss.data.item(),
                                                                                                                          smoothLoss.data.item()))
    writer.add_scalars("TrainLoss", {"totalLoss": totalLoss, "s2loss": totalS2Loss}, e)
    if (e + 1) % 5 == 0:
      torch.save(model.state_dict(), './outputs/model/{}-{}.pth'.format(saveModelName, e))
    # validation
    model.eval()
    with torch.no_grad():
      rmse = 0.0
      countV = 0
      for i, valBatchData in enumerate(valDataLoader):
        leftRgbS2 = valBatchData['leftRGBS2']
        rightRgbS2 = valBatchData['rightRGBS2']
        leftRgb = valBatchData['leftRGB']
        rightRgb = valBatchData['rightRGB']

        b, c, h, w = leftRgb.size()
        input = torch.cat([leftRgbS2, rightRgbS2], dim=1).to(device)
        inputImg = torch.cat([leftRgb, rightRgb], dim=1).to(device)
        leftDepthGt = valBatchData['leftDepth'].to(device)
        invalidMask = ((leftDepthGt > maxDepth) | (leftDepthGt <= 0) | torch.isnan(leftDepthGt))

        _, erp, predDep = model(input, inputImg)

        predDep[invalidMask] = 0.0
        erp[invalidMask] = 0.0
        leftDepthGt[invalidMask] = 0.0
        curRmse = torch.sum(torch.sqrt(torch.sum((predDep - leftDepthGt)**2, dim=[1, 2, 3]) / (torch.sum((~invalidMask).float(), dim=[1, 2, 3]))))
        rmse += curRmse.data.item()
        countV += b
        if i == 0:
          io.saveValImage(leftDepthGt, erp, predDep, e, saveImgDir)
      rmse /= countV
      valEndTime = time.time()
      print("epoch: {}, val time cost: {} s, rmse is {}".format(e, valEndTime - trainEndTime, rmse))
      writer.add_scalars("TrainLoss", {"rmse": rmse}, e)

    # adjust learing rate
    scheduler.step()

  torch.save(model.state_dict(), './outputs/model/{}-{}.pth'.format(saveModelName, e))


if __name__ == '__main__':
  main()
