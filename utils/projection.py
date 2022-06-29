import os
import numpy as np
import pickle

import torch


def batchERP(inputTensor, meshLevel, interMap, outSize=(256, 512)):
  """
  return: ERP depth map interpolate from s2 depth signal, with a shpae of NxCxHxW

  inputTensor: batch*channel*num tensor, the output of spherical cnns

  meshLevel: integer in [0,7] the level of mesh

  interMap: 1d tensor to indentify the interpolation index

  outSize: tuple, output image size, H,W for the 4-dim tensor(keep N,C as input)
  """
  b, ch, n = inputTensor.shape
  assert (ch == 1 or ch == 3 or ch == 4)
  outS = [b, ch, outSize[0] * outSize[1]]
  # img = torch.zeros(outS).cuda()
  inter = interMap.repeat(b, ch, 1)
  img = torch.gather(inputTensor, 2, inter)
  img = img.view([b, ch, outSize[0], outSize[1]]).cuda()
  return img
