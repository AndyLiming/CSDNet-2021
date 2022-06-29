import torch
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from torch.utils.data import Dataset
import pickle, gzip

import functools


def sparse2tensor(m):
  """
    Convert sparse matrix (scipy.sparse) to tensor (torch.sparse)
    """
  assert (isinstance(m, sparse.coo.coo_matrix))
  i = torch.LongTensor([m.row, m.col])
  v = torch.FloatTensor(m.data)
  return torch.sparse.FloatTensor(i, v, torch.Size(m.shape))


def spmatmul(den, sp):
  """
    den: Dense tensor of shape batch_size x in_chan x #V
    sp : Sparse tensor of shape newlen x #V
    """
  batch_size, in_chan, nv = list(den.size())
  new_len = sp.size()[0]
  den = den.permute(2, 1, 0).contiguous().view(nv, -1)
  res = torch.spmm(sp, den).view(new_len, in_chan, batch_size).contiguous().permute(2, 1, 0)
  return res


def xyz2latlong(vertices):
  x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
  long = np.arctan2(y, x)
  xy2 = x**2 + y**2
  lat = np.arctan2(z, np.sqrt(xy2))
  return lat, long


def interp_r2tos2(sig_r2, V, method="linear", dtype=np.float32):
  """
    sig_r2: rectangular shape of (lat, long, n_channels)
    V: array of spherical coordinates of shape (n_vertex, 3)
    method: interpolation method. "linear" or "nearest"
    """
  ele, azi = xyz2latlong(V)
  nlat, nlong = sig_r2.shape[0], sig_r2.shape[1]
  dlat, dlong = np.pi / (nlat - 1), 2 * np.pi / nlong
  lat = np.linspace(-np.pi / 2, np.pi / 2, nlat)
  long = np.linspace(-np.pi, np.pi, nlong + 1)
  sig_r2 = np.concatenate((sig_r2, sig_r2[:, 0:1]), axis=1)
  intp = RegularGridInterpolator((lat, long), sig_r2, method=method)
  s2 = np.array([ele, azi]).T
  sig_s2 = intp(s2).astype(dtype)
  return sig_s2


def greater(x, y):
  if (abs(x[0] - y[0]) > 0.00001 and x[0] > y[0]) or (abs(x[0] - y[0]) <= 0.00001 and abs(x[1] - y[1]) > 0.00001 and x[1] > y[1]):
    return True
  else:
    return False


def myComp(x, y):
  if greater(x, y):
    return 1
  elif abs(x[0] - y[0]) <= 0.00001 and abs(x[1] - y[1]) <= 0.00001:
    return 0
  else:
    return -1


def interp_s2tor2(sig_s2, V, method="linear", dtype=np.float32, destSize=(256, 512)):
  """
    sig_s2:
    V: array of spherical coordinates of shape (n_vertex, 3)
    method: interpolation method. "linear" or "nearest"
    """
  ele, azi = xyz2latlong(V)
  s2 = np.array([ele, azi]).T
  print("azi", np.min(azi), np.max(azi))
  print("ele", np.min(ele), np.max(ele))
  print(ele.shape, azi.shape)
  nlat, nlong = destSize[0], destSize[1]
  dlat, dlong = np.pi / (nlat - 1), 2 * np.pi / nlong
  lat = np.linspace(-np.pi / 2, np.pi / 2, nlat)
  long = np.linspace(-np.pi, np.pi, nlong)
  ss = np.concatenate((s2, sig_s2), axis=1)
  print(ss.shape)
  ss = sorted(ss, key=functools.cmp_to_key(myComp))
  ss = np.array(ss)
  np.savetxt('ss.txt', ss, delimiter=',')
  # print(ss)
  # sig = ss[:, 2:]
  # print(sig.shape)
  # out = sig.reshape(282, 581, 3)
  # return out
  # grid = np.zeros((destSize[0] * destSize[1], 2))
  # for i in range(destSize[0]):
  #   for j in range(destSize[1]):
  #     grid[i * destSize[1] + j][0] = lat[i]
  #     grid[i * destSize[1] + j][1] = long[j]
  # grid = grid.T
  # #sig_r2 = np.concatenate((sig_r2, sig_r2[:, 0:1]), axis=1)
  # #print(sig_r2.shape)
  # intp = RegularGridInterpolator(s2, sig_s2, method=method)
  # sig_r2 = intp(grid).astype(dtype)
  # print("sig_r2.shape :{}".format(sig_r2.shape))
  return sig_s2


class MNIST_S2_Loader(Dataset):
  """Data loader for spherical MNIST dataset."""
  def __init__(self, data_zip, partition="train"):
    """
        Args:
            data_zip: path to zip file for data
            partition: train or test
        """
    assert (partition in ["train", "test"])
    self.data_dict = pickle.load(gzip.open(data_zip, "rb"))
    if partition == "train":
      self.x = self.data_dict["train_inputs"] / 255
      self.y = self.data_dict["train_labels"]
    else:
      self.x = self.data_dict["test_inputs"] / 255
      self.y = self.data_dict["test_labels"]
    self.x = (np.expand_dims(self.x, 1) - 0.1307) / 0.3081

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]


if __name__ == '__main__':
  import cv2
  imgName = '../../tmp/ca2e_011137_12_1.png'
  meshLevel = 7
  img = cv2.imread(imgName, cv2.IMREAD_COLOR)
  print(img.shape)
  meshFile = pickle.load(open('../meshfiles/icosphere_{}.pkl'.format(meshLevel), 'rb'))
  V = meshFile['V']
  s2 = interp_r2tos2(img, V)
  np.savetxt('leftImgCarla-{}.txt'.format(meshLevel), s2)
