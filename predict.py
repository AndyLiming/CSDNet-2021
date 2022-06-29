import argparse
import os
import sys
import numpy as np
import pickle
import time

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

import models
from dataset import Dataset360D
from utils import loss, io, projection
from models import CSDNet
import spherical as S360

from PIL import Image

sys.path.append("../")
from meshcnn.utils import xyz2latlong, interp_r2tos2

epsilon = 1e-10


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--leftImg', default='./outputs/left.jpg', type=str, help="input left image name")
  parser.add_argument('--rightImg', default='./outputs/right.jpg', type=str, help="input right image name")
  parser.add_argument('--model_dir', default='./outputs/model/', type=str, help="directory to save model files")
  parser.add_argument('--load_ckpt', default=None, type=str, help="model file name")
  parser.add_argument('--save_dir', default='./outputs/', type=str, help="directory to save output images")
  parser.add_argument('--width', default=512, type=int, help="width of input and output images")
  parser.add_argument('--height', default=256, type=int, help="height of input and output images")
  parser.add_argument('--channel', default=3, type=int, help="channels of input image, usually set 3 for RGB image and 1 for gray")
  parser.add_argument('--mesh_level', default=6, type=int, help="mesh level of spherical CNNs")
  parser.add_argument('--max_depth', default=20.0, type=float, help="max valid depth")
  parser.add_argument('--baseline', default=0.26, type=float, help="baseline of binocular spherical system")
  parser.add_argument('--parallel', action='store_true', help="if use data parallel or not")

  args = parser.parse_args()

  maxDepth = args.max_depth
  meshLevel = args.mesh_level
  width, height = args.width, args.height
  channel = args.channel

  meshFile = pickle.load(open('./meshfiles/icosphere_{}.pkl'.format(meshLevel), 'rb'))
  V = meshFile['V']

  leftRGB = Image.open(args.leftImg).convert('RGB')
  rightRGB = Image.open(args.rightImg).convert('RGB')
  leftRGBS2 = interp_r2tos2(np.array(leftRGB), V)
  rightRGBS2 = interp_r2tos2(np.array(rightRGB), V)
  leftRGBS2 = leftRGBS2.T
  rightRGBS2 = rightRGBS2.T
  leftRGBS2 = torch.from_numpy(leftRGBS2)
  rightRGBS2 = torch.from_numpy(rightRGBS2)

  trans = transforms.Compose([transforms.Resize([256, 512]), transforms.ToTensor()])
  leftRGB = trans(leftRGB)
  rightRGB = trans(rightRGB)

  leftRGBS2.unsqueeze_(0)
  rightRGBS2.unsqueeze_(0)
  leftRGB.unsqueeze_(0)
  rightRGB.unsqueeze_(0)
  saveDir = args.save_dir
  saveName = os.path.join(args.save_dir, 'predictDepth.png')

  dense = True  # equirectangular projection method

  meshVerticesMap = {0: 12, 1: 42, 2: 162, 3: 642, 4: 2562, 5: 10242, 6: 40962, 7: 163842}

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
  # parallel
  if args.parallel:
    model = nn.DataParallel(model)
  model.to(device)

  #height, width = width, height
  if dense:
    interMap = np.load('./interMapFiles/interMap_{}-1d.npy'.format(meshLevel)).astype(np.int64)
  else:
    interMap = np.load('./interMapFiles/interMap_{}-1s.npy'.format(meshLevel)).astype(np.int64)
  interMap = torch.from_numpy(interMap).to(device)

  counter = 0
  model.eval()
  with torch.no_grad():
    input = torch.cat([leftRGBS2, rightRGBS2], dim=1).to(device)
    inputImg = torch.cat([leftRGB, rightRGB], dim=1).to(device)
    print(input.shape, inputImg.shape)
    predS2, erp, predDep = model(input, inputImg)
    predDep[predDep < epsilon] = epsilon
    erp[erp < epsilon] = epsilon
    predS2[predS2 < epsilon] = epsilon
    predSave = predDep
    predSave = (predSave - torch.min(predSave)) / (torch.max(predSave) - torch.min(predSave))
    torchvision.utils.save_image(predSave, saveName)


if __name__ == "__main__":
  main()
