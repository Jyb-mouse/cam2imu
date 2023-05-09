import os
import json
import geojson
import pygame
import pyproj
import math
import cv2 as cv
import numpy as np
from pyproj import Transformer

from transformation_utils import ENUTransformer
from transformation_utils import ECEFTransformer
from transformation_utils import TransformPoints,euler2rotation
from scipy.spatial.transform import Rotation as R
from bag_reader import RosbagReader



class Map_server(object):

    def __init__(self, geojsonfile_path):
        self.geojsonfile_path = geojsonfile_path

    def read_geojson(self):
        file_path = self.geojsonfile_path
        features = []
        with open(file_path) as f:
            gj = geojson.load(f)
            for i in gj['features']:
                features.append(i)
            # print(features)
        return features

    def get_points(self):
        features = self.read_geojson()
        mark = []
        for feature in features:
            for i in feature['geometry']['coordinates']:
                for j in i:  
                    mark.append(j)
        mark = np.array(mark)
        return mark

    def trans_points_to_imu_coor(self, enu_points, T_enu2imu):
        points_org = enu_points
        trans_mat = T_enu2imu
        points_with_imu = []
        for i in points_org:
            for j in i:
                cur_point = np.array([j[0], j[1], j[2]])
                cur_point_with_imu = TransformPoints(cur_point, trans_mat)
                points_with_imu.append(cur_point_with_imu)
        return points_with_imu
    
    def lla2enu(self, gps_info):
        cur_gps_info = gps_info
        cur_lla = np.array([cur_gps_info.latitude, cur_gps_info.longitude, cur_gps_info.altitude])
        transformertest1 = pyproj.Transformer.from_crs(
            "epsg:4326", "epsg:32651"
            )
        x,y,z = transformertest1.transform(cur_gps_info.latitude, cur_gps_info.longitude, cur_gps_info.altitude)
        enu_imu = np.array([x,y,z])                
        return enu_imu

    def get_utm_yaw_bias(self, lat, lon, utm_zone):
        lon_diff_rad = np.deg2rad(lon - (6*utm_zone-183))
        lat_rad = np.deg2rad(lat)
        diff = np.rad2deg(math.atan(math.tan(lon_diff_rad)*math.sin(lat_rad)))
        return diff

    def geodetic_to_enu_rot(self, roll, pitch, yaw):
        cos_roll = math.cos(roll)
        cos_pitch = math.cos(pitch)
        cos_yaw = math.cos(yaw)
        sin_roll = math.sin(roll)
        sin_pitch = math.sin(pitch)
        sin_yaw = math.sin(yaw) 

        p_0_0 = cos_yaw * cos_roll - sin_yaw * sin_pitch * sin_roll
        p_1_0 = sin_yaw * cos_roll + cos_yaw * sin_pitch * sin_roll
        p_2_0 = -cos_pitch * sin_roll
        p_0_1 = -sin_yaw * cos_pitch
        p_1_1 = cos_yaw * cos_pitch
        p_2_1 = sin_pitch
        p_0_2 = cos_yaw * sin_roll + sin_yaw * sin_pitch * cos_roll
        p_1_2 = sin_yaw * sin_roll - cos_yaw * sin_pitch * cos_roll
        p_2_2 = cos_pitch * cos_roll
        body_enu = np.matrix([p_0_0,p_0_1,p_0_2,p_1_0,p_1_1,p_1_2,p_2_0,p_2_1,p_2_2]).reshape(3,3)
        return body_enu

    def get_R_imu2enu(self, gps_info):
        cur_gps_info = gps_info
        utm_zone = 51
        diff = self.get_utm_yaw_bias(cur_gps_info.latitude, cur_gps_info.longitude, utm_zone)
        # R_imu2enu =  R.from_euler('xyz', [cur_gps_info.position_covariance[2]  ,cur_gps_info.position_covariance[1],
        #                  -cur_gps_info.position_covariance[3]], degrees=True).as_matrix()
        a = np.deg2rad(np.array([cur_gps_info.position_covariance[1],cur_gps_info.position_covariance[2], -cur_gps_info.position_covariance[3]] + diff))
        R_imu2enu = self.geodetic_to_enu_rot(a[0], a[1],a[2])
        print(R_imu2enu)
        return R_imu2enu

    def get_T_enu2imu(self, R_imu2enu, t_imu2enu):
        R1 = R_imu2enu
        t1 = t_imu2enu
        T_imu2enu = np.identity(4)
        T_imu2enu[:3,:3] = R1
        T_imu2enu[:3, 3] = t1
        T_enu2imu = np.linalg.inv(T_imu2enu)
        return T_enu2imu

    def transPoints_enu2imu(self, map_marks, T_enu2imu):
        points = map_marks
        T = T_enu2imu
        # points_imu = []
        # for i in points:
        #     point_ori = [i[0],i[1],i[2]]
        #     point_imu = TransformPoints(point_ori, T)
        #     points_imu.append(point_imu)
        points_imu = TransformPoints(map_marks, T_enu2imu)
        return points_imu

def main():

    file_path = './marking.geojson'
    map_server = Map_server(file_path)

    mark = map_server.get_points()
    # print(mark)
    # print(len(mark[8]))

    transfor = ENUTransformer()
    transfor.set_base([31.206388889, 121.689444445])
    lat_org = 31.206388889 # deg
    lon_org = 121.689444445  # deg
    alt_org = 1673     # meters

    # print(mark[0][0])
    transformertest1 = pyproj.Transformer.from_crs(
            "epsg:4326", "epsg:32651"
            )
    transformertest2 = pyproj.Transformer.from_crs(
            "epsg:32651","epsg:4326"
            )
    point_org = np.array(mark[0][0][0], mark[0][0][1], mark[0][0][2])
    x_org, y_org, z_org = transformertest2.transform(mark[0][0][0], mark[0][0][1], mark[0][0][2])
    print(x_org, y_org, z_org)

    bag_file = '/home/mouse/Documents/cam2imu_manual_calib/src/cam2imu_manual_calib/cam2imu_manual_calib/total.bag'
    ts_begin = '00:00:00'
    ts_end = '00:00:05'
    base = [31.206388889, 121.689444445]
    aligner = '/camera1/image_raw/compressed'
    topics = {    
        'gps-0': '/met/gps/lla_qj',
        'image': '/camera1/image_raw/compressed',
        'velo': '/met/gps/velocity'
    }

    bag_reader = RosbagReader(bag_file, base)
    bag_reader.init_topics(aligner, topics)
    for data in bag_reader.read(ts_begin, ts_end):
        print(data['indix'])
        print("GPS_info:------------------------------")
        # print(data['gps-0'][1])
        lla_qj = np.array([data['gps-0'][1].latitude, data['gps-0'][1].longitude, data['gps-0'][1].altitude])
        x,y,z = transformertest1.transform(data['gps-0'][1].latitude, data['gps-0'][1].longitude, data['gps-0'][1].altitude)
        enu_imu = np.array([x,y,z])
        print('point_ENU:',point_org)
        print('enu_imu:',enu_imu)
        R_imu2enu =  R.from_euler('zxy', [-data['gps-0'][1].position_covariance[3],data['gps-0'][1].position_covariance[1], data['gps-0'][1].position_covariance[2]], degrees=True).as_matrix()
        print(R_imu2enu)
        T_imu2enu = np.identity(4)
        T_imu2enu[:3,:3] = R_imu2enu
        T_imu2enu[:3, 3] = enu_imu
        T_enu2imu = np.linalg.inv(T_imu2enu)

        print('T :', T_enu2imu)
        pose_imu = TransformPoints(point_org, T_enu2imu)
        print('point_imu:',pose_imu)

        # break
        # print("Image_info-------------")
        # print((data['image'][0]))


if __name__ == "__main__":
    main()