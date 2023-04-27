import math
import numpy as np
from copy import deepcopy

# rad
def euler2rotation(roll, pitch, yaw):
    rx = np.matrix([
        1, 0, 0, 
        0, math.cos(roll), -math.sin(roll), 
        0, math.sin(roll), math.cos(roll)
    ]).reshape(3,3)
    ry = np.matrix([
        math.cos(pitch), 0, math.sin(pitch), 
        0, 1, 0, 
        -math.sin(pitch), 0, math.cos(pitch)
    ]).reshape(3,3)
    rz = np.matrix([
        math.cos(yaw), -math.sin(yaw), 0, 
        math.sin(yaw), math.cos(yaw), 0, 
        0, 0, 1
    ]).reshape(3,3)
    rot = np.dot(rz, np.dot(ry, rx))
    return rot

def rotation2euler(rot):
    roll = math.atan2(rot[2,1], rot[2,2])
    pitch = math.asin(-rot[2,0])
    yaw = math.atan2(rot[1,0], rot[0,0])
    return np.array([roll, pitch, yaw])

# rad: roll pitch yaw 
def RT2Transmat(angle, xyz):
    rot = euler2rotation(angle[0], angle[1], angle[2])
    trans_mat = np.identity(4)
    trans_mat[0:3, 0:3] = rot
    trans_mat[0:3, 3] = xyz
    return trans_mat

def Transmat2RT(mat):
    angle = rotation2euler(mat[0:3, 0:3])
    xyz = mat[0:3, 3]
    return angle, xyz

def TransformPoints(points, trans_mat):
    input_points = deepcopy(points)
    rot = trans_mat[0:3, 0:3]
    xyz = trans_mat[0:3, 3]
    res = np.dot(rot, input_points.T).T + xyz
    return res



if __name__ == '__main__':
    roll, pitch, yaw = np.deg2rad(0.355), np.deg2rad(-0.149), np.deg2rad(-342.265)

    x,y,z = -1924.31472163,3027.86285561,16.37766588

    points = np.array([[ 2,  4,  6],
       [ 5,  7,  9],
       [ 8, 10, 12]])

    trans_mat =  RT2Transmat(np.array([roll, pitch, yaw]), np.array([x,y,z]))
    trans_points = TransformPoints(points, trans_mat)

    R, T = Transmat2RT(trans_mat)
    print(trans_mat)
    print(np.rad2deg(R[0]),np.rad2deg(R[1]),np.rad2deg(R[2]))
    print(T)
    print(trans_points)

