import sys
import os
import numpy as np
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
import pickle, gzip

from meshcnn.utils import xyz2latlong


def find_nearest(array, value):
  array = np.asarray(array)
  diff = np.abs(array - value)
  idx = ((diff[:, 0]) + (diff[:, 1])).argmin()
  #idx = np.sqrt((diff[:, 0]**2 + diff[:, 1]**2)).argmin()
  return idx


def gen_intermap(height, width, meshLevel):
  meshFile = pickle.load(open('./meshfiles/icosphere_{}.pkl'.format(meshLevel), 'rb'))
  V = meshFile['V']
  ele, azi = xyz2latlong(V)  # lat, long
  # lat = np.linspace(-np.pi / 2, np.pi / 2, height+1).astype(np.float32)
  # long = np.linspace(-np.pi, np.pi, width + 1).astype(np.float32)
  lat = ((np.linspace(0, height - 1, height)) * np.pi / height - np.pi / 2).astype(np.float32)
  long = ((np.linspace(0, width - 1, width)) * 2 * np.pi / width - np.pi).astype(np.float32)
  s2 = np.array([ele, azi]).T.astype(np.float32)

  res = np.zeros(height * width)

  count = 0
  for i in range(height):
    for j in range(width):
      pos = np.array([lat[i], long[j]])
      id = find_nearest(s2, pos)
      res[count] = id
      # if res[count] != intermap_6[count]:
      #   print(i, j, res[count], s2[int(res[count])], intermap_6[count], s2[intermap_6[count]])
      count += 1
  return res


if __name__ == '__main__':
  interMapDir = './interMapFiles'
  os.makedirs(interMapDir, exist_ok=True)
  height, width = 256, 512
  mesh_level = 6
  res = gen_intermap(height, width, mesh_level).astype(np.int64)
  np.save(os.path.join(interMapDir, 'interMap_{}-1d.npy'.format(mesh_level)), res)
