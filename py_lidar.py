import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import ros_numpy

from sklearn import preprocessing
from struct import *

offset = 0

def normalize(data):
    _range = np.max(data) - np.min(data)
    if _range==0:
        return (data - np.min(data))
    else:
        return (data - np.min(data)) / _range

def trans_pointcloud2_to_array(pc_msg, remove_nans=True, dtype=np.float):
    cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)

    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']

    info = np.zeros(cloud_array.shape + (3,), dtype=dtype)
    info[...,0] = cloud_array['intensity']
    info[...,1] = cloud_array['ring']
    info[...,2] = cloud_array['timestamp']*1e9

    return points, info

def save_pcd_with_data(file_path, point_array, info):
    if point_array is None:
        print('empty points array')
        return
    if len(point_array) == 0:
        print('empty points array')
        return
    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud(device)
    
    xyz = point_array
    intensity = info[:, 0]
    colors = intensity2rgb(intensity, 0.98)
    scalar = [[i] for i in intensity]

    pcd.point.positions = o3d.core.Tensor(xyz, dtype, device)
    pcd.point.colors = o3d.core.Tensor(colors, dtype, device)
    pcd.point.intensities = o3d.core.Tensor(scalar, dtype, device)

    o3d.t.io.write_point_cloud(file_path, pcd)

def intensity2rgb(intensity:np.ndarray, rat):

    def adjust_display_range_ratio(data, ratio):
        max_intensity = np.max(data)
        threshold = max_intensity * ratio
        data[data > threshold] = threshold
        return data

    intensity = adjust_display_range_ratio(intensity, rat)
    scalar = normalize(intensity)
    cmap = plt.get_cmap('gist_ncar')

    rgba_img = cmap(scalar)
    rgb_img = np.delete(rgba_img, 3, 1)

    return np.squeeze(rgb_img)

def save_lidar_binary(ts, data, info, lidar_file, idx_file):
    global offset
    point_size = len(data)
    for i in range(point_size):
        buffer_data = pack(
            'fffff',
            data[i][0],  # x
            data[i][1],  # y
            data[i][2],  # z
            info[i][0],  # intensity
            info[i][1]  # ring/beam id
        )
        lidar_file.write(buffer_data)
    length = len(buffer_data) * point_size
    buffer_idx = str(ts) + ' ' + \
        str(offset) + ' ' + str(length)
    buffer_idx += '\n'
    idx_file.write(buffer_idx)
    offset += length



    

