'''
Author: jin yanbin
email: jinyanbin@maxieyetech.com
Date: 2023-04-06 10:14:10
LastEditTime: 2023-04-10 18:12:50
filePath: Do not edit
'''
import numpy as np
import cv2 as cv
from calib_utils.transformation_utils.transform_utils import Transmat2RT, RT2Transmat, euler2rotation, rotation2euler

def get_T_lidar2vehicle1(delta_rpy, xyz, T_lidar2cam):
    cam2veh_rpy = np.array((delta_rpy[0] - 90, delta_rpy[1], -delta_rpy[2] - 90))
    cam2veh_v = np.deg2rad(cam2veh_rpy)
    R_cam2vehicle1 = euler2rotation(cam2veh_v[0], cam2veh_v[1], cam2veh_v[2]) 
    T_cam2vehicle1 = RT2Transmat(cam2veh_v, xyz)
    print('T_cam2vehicle1 = ', T_cam2vehicle1)
    T_vehicle1_2_vehicle2 = np.array(
                                [[1, 0,  0,  3.6],
                                [ 0, 1, 0, 0], 
                                [ 0, 0, 1, 0],
                                [ 0.,  0.,  0.,  1.]])
    T_cam2vehicle2 = np.dot(T_vehicle1_2_vehicle2, T_cam2vehicle1)
    print('T_cam2vehicle2 = ', T_cam2vehicle2)
    T_vehicle2cam = np.linalg.inv(T_cam2vehicle2)
    print('T_vehicle2_2_cam = ', T_vehicle2cam)
    T_lidar2vehicle2 = np.dot(T_cam2vehicle2, T_lidar2cam)
    return T_lidar2vehicle2, T_vehicle2cam

if __name__ == "__main__":
    # NEW_gs4
    T_lidar2cam = np.array([0.9985790415454673, 0.047009502655825215, -0.02509988936603994, -0.05231122671149021,
                             -0.026086567296978086, 0.020494267444703864, -0.9994495865267868, -0.8057127872194237,
                             -0.046469224146694306,  0.9986841801400264,  0.021691462496737514, -1.297756962276894,
                             0.0, 0.0, 0.0, 1.0]).reshape(4,4)
    
    cam2veh_rpy = np.array((-1.27187, 0, 0.84032))
    xyz = np.array((-1.4, 0, 1.68))
    T_lidar2vehicle2, T_vehicle2cam = get_T_lidar2vehicle1(cam2veh_rpy, xyz, T_lidar2cam)
    
    T_lidar2lidar = np.dot(np.linalg.inv(T_lidar2cam), np.dot(T_vehicle2cam, T_lidar2vehicle2))
    
    # T_lidar2vehicle2 = np.dot(T_vehicle1_2_vehicle2, T_lidar2vehicle1)
    print('T_vehicle2cam= ', T_vehicle2cam)
    print('T_lidar2vehicle2 = ', T_lidar2vehicle2)
    # print('T_vehicle2cam= ', T_vehicle2cam)
    print('T_lidar2lidar = ', T_lidar2lidar)

