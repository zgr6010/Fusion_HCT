import h5py
import numpy as np
from scipy import io
mat = h5py.File('../data/Trento_Data/Lidar_Trento.mat', 'r+')
print(mat.keys())
print(mat.values())
f = mat['lidar_trento']
mat_t = np.transpose(f)

io.savemat('data/Lidar_Trento.mat', {'lidar_trento': mat_t})
# print(mat['GT_houston'].shape)