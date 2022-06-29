import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle

# import meshcnn
sys.path.append("../")
from meshcnn.ops import MeshConv, MeshConv_transpose, ResBlock
from utils import projection


def xyz2latlong(vertices):
  x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
  long = np.arctan2(y, x)
  xy2 = x**2 + y**2
  lat = np.arctan2(z, np.sqrt(xy2))
  return lat, long


class Down(nn.Module):
  def __init__(self, in_ch, out_ch, level, mesh_folder, bias=True):
    """
        use mesh_file for the mesh of one-level down
        """
    super().__init__()
    self.conv = ResBlock(in_ch, in_ch, out_ch, level + 1, True, mesh_folder)

  def forward(self, x):
    x = self.conv(x)
    return x


class Up(nn.Module):
  def __init__(self, in_ch, out_ch, level, mesh_folder, bias=True):
    """
        use mesh_file for the mesh of one-level up
        """
    super().__init__()
    mesh_file = os.path.join(mesh_folder, "icosphere_{}.pkl".format(level))
    self.up = MeshConv_transpose(in_ch, in_ch, mesh_file, stride=2)
    self.conv = ResBlock(in_ch, out_ch, out_ch, level, False, mesh_folder)

  def forward(self, x):
    x = self.up(x)
    x = self.conv(x)
    return x


class PlaneResBlock(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.conv = nn.Sequential(nn.Conv2d(dim,
                                        dim,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1),
                              nn.BatchNorm2d(dim),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(dim,
                                        dim,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1),
                              nn.BatchNorm2d(dim))
    self.act = nn.ReLU(inplace=True)

  def forward(self, x):
    out = x + self.conv(x)
    out = self.act(out)
    return out


# refine residual net
# input: left + right + ERP_depth
# output: refined depth map


class RefineNet(nn.Module):
  def __init__(self, in_ch, out_ch=1, fdim=16, nDown=1, nRes=3):
    super().__init__()
    self.inConv = nn.Sequential(nn.Conv2d(in_ch, fdim, kernel_size=7, stride=1, padding=3, bias=False), nn.BatchNorm2d(fdim), nn.ReLU(inplace=True))
    self.down = []
    self.res = []
    self.up = []
    for i in range(nDown):
      self.down.append(nn.Conv2d(fdim * (2**(i)), fdim * (2**(i + 1)), kernel_size=3, stride=2, padding=1, bias=False))
      self.down.append(nn.BatchNorm2d(fdim * (2**(i + 1))))
      self.down.append(nn.ReLU(inplace=True))

    for i in range(nRes):
      self.res.append(PlaneResBlock(fdim * (2**(nDown))))

    for i in range(nDown):
      self.up.append(nn.ConvTranspose2d(fdim * (2**(nDown - i)), fdim * (2**(nDown - i - 1)), kernel_size=4, stride=2, padding=1, bias=False))
      self.up.append(nn.BatchNorm2d(fdim * (2**(nDown - i - 1))))
      self.up.append(nn.ReLU(inplace=True))

    self.outConv = nn.Sequential(nn.Conv2d(fdim, out_ch, kernel_size=3, stride=1, padding=1, bias=False))

    self.down = nn.Sequential(*self.down)
    self.res = nn.Sequential(*self.res)
    self.up = nn.Sequential(*self.up)

  def forward(self, x):
    out = self.inConv(x)
    out = self.down(out)
    out = self.res(out)
    out = self.up(out)
    out = self.outConv(out)
    return out


# cascade spherical depth networks
class CSDNet(nn.Module):
  def __init__(self,
               in_ch,
               out_ch,
               inSize=(256,
                       512),
               max_level=5,
               min_level=0,
               fdim=16,
               resNum=3,
               refineDownNum=1,
               refineResNum=3,
               refineFdim=16,
               mesh_folder='./meshfiles/',
               dense=True,
               parallel=False):
    super().__init__()
    self.mesh_folder = mesh_folder
    self.fdim = fdim
    self.max_level = max_level
    self.min_level = min_level
    self.levels = max_level - min_level
    self.resNum = resNum
    self.dense = dense
    self.down = []
    self.res = []
    self.up = []
    self.in_conv = MeshConv(in_ch, fdim, self.__meshfile(max_level), stride=1)
    self.out_mesh_conv = MeshConv(fdim, fdim, self.__meshfile(max_level), stride=1)
    self.out_conv = nn.Sequential(nn.ReLU(inplace=True), nn.Conv1d(fdim, out_ch, kernel_size=1, bias=False))

    # Downsamp path
    for i in range(self.levels - 1):
      self.down.append(Down(fdim * (2**i), fdim * (2**(i + 1)), max_level - i - 1, mesh_folder))
    self.down.append(Down(fdim * (2**(self.levels - 1)), fdim * (2**(self.levels - 1)), min_level, mesh_folder))

    # Res block
    for i in range(self.resNum):
      self.res.append(
          ResBlock(in_chan=fdim * (2**(self.levels - 1)),
                   neck_chan=fdim * (2**(self.levels - 1)),
                   out_chan=fdim * (2**(self.levels - 1)),
                   level=self.min_level,
                   coarsen=False,
                   mesh_folder=self.mesh_folder))

    # Upsamp path
    for i in range(self.levels - 1):
      self.up.append(Up(fdim * (2**(self.levels - i - 1)), fdim * (2**(self.levels - i - 2)), min_level + i + 1, mesh_folder))
    self.up.append(Up(fdim, fdim, max_level, mesh_folder))

    self.down = nn.Sequential(*self.down)
    self.res = nn.Sequential(*self.res)
    self.up = nn.Sequential(*self.up)
    self.refine = RefineNet(in_ch + out_ch, out_ch=1, fdim=refineFdim, nDown=refineDownNum, nRes=refineResNum)

    interMap = np.load('./interMapFiles/interMap_{}-1d.npy'.format(max_level)).astype(np.int64)

    self.parallel = parallel

    self.interMap = torch.from_numpy(interMap)
    if not self.parallel:
      self.interMap = self.interMap.cuda()
    #print(self.interMap.shape, self.interMap.dtype)

  def forward(self, x, ximg):
    x = self.in_conv(x)
    x = self.down(x)
    x = self.res(x)
    x = self.up(x)
    x = self.out_mesh_conv(x)
    x = self.out_conv(x)
    x = torch.abs(x)  #assure postive output
    #pro = projection.batchERP(x, self.max_level, self.interMap, dense=self.dense)
    if self.parallel:
      pro = projection.batchERP(x.cpu(), self.max_level, self.interMap)
    else:
      pro = projection.batchERP(x, self.max_level, self.interMap)
    if pro.shape != ximg.shape:
      pro = F.interpolate(pro, size=(ximg.shape[2], ximg.shape[3]))
    x1 = torch.cat([ximg, pro], dim=1)
    x1 = self.refine(x1)
    return x, pro, x1

  def __meshfile(self, i):
    return os.path.join(self.mesh_folder, "icosphere_{}.pkl".format(i))


# cascade plane CSDNet
class CSDNet_plane(nn.Module):
  def __init__(self, in_ch, out_ch, inSize=(256, 512), max_level=5, min_level=0, fdim=16, resNum=3, refineDownNum=1, refineResNum=3, refineFdim=16, mesh_folder='./meshfiles/'):
    super().__init__()
    self.mesh_folder = mesh_folder
    self.fdim = fdim
    self.max_level = max_level
    self.min_level = min_level
    self.levels = max_level - min_level
    self.resNum = resNum
    self.down = []
    self.res = []
    self.up = []
    self.in_conv = nn.Conv2d(in_ch, fdim, kernel_size=3, stride=1, padding=1, bias=False)
    self.out_conv = nn.Conv2d(fdim, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

    # Downsamp path
    for i in range(self.levels - 1):
      self.down.append(nn.Conv2d(fdim * (2**i), fdim * (2**(i + 1)), kernel_size=3, stride=2, padding=1, bias=False))
      self.down.append(nn.BatchNorm2d(fdim * (2**(i + 1))))
      self.down.append(nn.ReLU(inplace=True))
    self.down.append(nn.Conv2d(fdim * (2**(self.levels - 1)), fdim * (2**(self.levels - 1)), kernel_size=3, stride=2, padding=1, bias=False))
    self.down.append(nn.BatchNorm2d(fdim * (2**(self.levels - 1))))
    self.down.append(nn.ReLU(inplace=True))

    # Res block
    for i in range(self.resNum):
      self.res.append(PlaneResBlock(fdim * (2**(self.levels - 1))))

    # Upsamp path
    for i in range(self.levels - 1):
      self.up.append(nn.ConvTranspose2d(fdim * (2**(self.levels - i - 1)), fdim * (2**(self.levels - i - 2)), kernel_size=4, stride=2, padding=1, bias=False))
      self.up.append(nn.BatchNorm2d(fdim * (2**(self.levels - i - 2))))
      self.up.append(nn.ReLU(inplace=True))
    self.up.append(nn.ConvTranspose2d(fdim, fdim, kernel_size=4, stride=2, padding=1, bias=False))
    self.up.append(nn.BatchNorm2d(fdim))
    self.up.append(nn.ReLU(inplace=True))

    self.down = nn.Sequential(*self.down)
    self.res = nn.Sequential(*self.res)
    self.up = nn.Sequential(*self.up)
    self.refine = RefineNet(in_ch + out_ch, out_ch=1, fdim=refineFdim, nDown=refineDownNum, nRes=refineResNum)

    interMap = np.load('./interMapFiles/interMap_{}-1d.npy'.format(max_level)).astype(np.int64)

    self.interMap = torch.from_numpy(interMap).cuda()

  def forward(self, ximg):
    x = self.in_conv(ximg)
    x = self.down(x)
    x = self.res(x)
    x = self.up(x)
    x = self.out_conv(x)
    x1 = torch.cat([ximg, x], dim=1)
    x1 = self.refine(x1)
    return x

  def __meshfile(self, i):
    return os.path.join(self.mesh_folder, "icosphere_{}.pkl".format(i))


def modelParaInit(model, type='xavier'):
  init_func = None
  if type == "xavier":
    init_func = torch.nn.init.xavier_normal_
  elif type == "kaiming":
    init_func = torch.nn.init.kaiming_normal_
  elif type == "gaussian" or type == "normal":
    init_func = torch.nn.init.normal_

  if init_func is not None:
    for module in model.modules():
      if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.ConvTranspose2d):
        init_func(module.weight)
        if module.bias is not None:
          module.bias.data.zero_()
      elif isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
  else:
    print("error when initialize the model's weights!")
