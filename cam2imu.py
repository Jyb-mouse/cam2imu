import os
import cv2 as cv
import numpy as np
import yaml
import pyproj
from pyquaternion import Quaternion
from pyproj import Transformer
from transformation_utils import TransformPoints
from scipy.spatial.transform import Rotation as R
from bag_reader import RosbagReader
from map_service import Map_server
from board import Board
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from config import ConfigLoader

class Cam2imu(object):
    def __init__(self, cfg):
        self.bag_path = cfg.bag_path
        self.ts_begin = cfg.ts_begin
        self.ts_end = cfg.ts_end
        self.base = cfg.base
        self.aligner = cfg.aligner
        self.topics = cfg.topics
        self.output_path = cfg.output_path
        self.cam_intrinsic = cfg.cam_intrinsic
        self.cam_distort = cfg.cam_distort
        self.new_camera_intrinsic = np.array([]) 

        self.map_server = Map_server(cfg.geojson_path)
        self.bag_reader = RosbagReader(cfg.bag_path, cfg.base)
        self.bag_reader.init_topics(self.aligner, self.topics)
        self.key_board = Board(cfg.output_path, cfg.cam_id, cfg.imu_id, cfg.type, cfg.width, cfg.height)


    def plot_point(self, points1, points2):
        X1, Y1, Z1=points1[:, 0],points1[:, 1],points1[:, 2]
        # X2, Y2, Z2=points2[:, 0],points2[:, 1],points2[:, 2]
        X2 = points2[0]
        Y2 = points2[1]
        plt.scatter(X1, Y1)
        plt.plot(X2, Y2, 'ro')

    def draw_point(self, img_pts, img):
        leng = len(img_pts)
        for i in range(0,leng ):
            cv.circle(img,(int(img_pts[i][0]),int(img_pts[i][1])),2,(0,0,255),-1)        
            if i % 5 == 0:
                if i < leng / 2:
                    cv.line(img, (int(img_pts[i][0]),int(img_pts[i][1])), (int(img_pts[i+1][0]),int(img_pts[i+1][1])), (0, 255, 0), 1)
                    cv.line(img, (int(img_pts[i+1][0]),int(img_pts[i+1][1])), (int(img_pts[i+2][0]),int(img_pts[i+2][1])), (0, 255, 0), 1)
                    cv.line(img, (int(img_pts[i+2][0]),int(img_pts[i+2][1])), (int(img_pts[i+3][0]),int(img_pts[i+3][1])), (0, 255, 0), 1)
                    cv.line(img, (int(img_pts[i+3][0]),int(img_pts[i+3][1])), (int(img_pts[i+4][0]),int(img_pts[i+4][1])), (0, 255, 0), 1)
        cv.imwrite('./new.jpg',img)

    def get_init_extrinsic(self, path):
        yaml_con = yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)
        cut_T_i2c = yaml_con['extrinsic']
        return cut_T_i2c

    def get_valid_point(self, points_imu, cut_T_i2c):
        points_finaly = TransformPoints(points_imu, cut_T_i2c)
        valid_idx = np.logical_and(points_finaly[:, 2] > 1.0, points_finaly[:, 2] < 100)
        points_finaly=points_finaly[valid_idx]
        return points_finaly

    def run(self, map_marks):
        cut_T_i2c = self.get_init_extrinsic(self.output_path)
        cut_T_i2c = np.array(cut_T_i2c)
        for data in self.bag_reader.read(self.ts_begin, self.ts_end):
            
            # get GPS and image information
            print(data['indix'])
            cur_gps_info = data['gps-0'][1]

            print("gps_timestamp: ", data['gps-0'][0]/1e9)
            print("image_timestamp: ", data['image'][0]/1e9)
            img_data = data['image'][1]
            img = cv.imdecode(np.fromstring(img_data.data,np.uint8), cv.IMREAD_COLOR)
            
            # get points in imu coordinary
            # lla_qj = np.array([data['gps-0'][1].latitude, data['gps-0'][1].longitude, data['gps-0'][1].altitude])
            enu_imu = self.map_server.lla2enu(cur_gps_info)
            R_imu2enu = self.map_server.get_R_imu2enu(cur_gps_info)
            # q = Quaternion(matrix = R_imu2enu)
            # print(q.x, q.y, q.z, q.w)
            T_enu2imu = self.map_server.get_T_enu2imu(R_imu2enu, enu_imu)
            points_imu = self.map_server.transPoints_enu2imu(map_marks, T_enu2imu)           
            print("current_T_i2c:", cut_T_i2c)

            # points_finalys = self.get_valid_point(points_imu, cut_T_i2c)
            # img_pts = im_utils.proj_cam2img(points_finalys, self.cam_intrinsic, self.cam_distort, self.new_camera_intrinsic, width=1824, height=944, cam_type=type)
            self.key_board.set_input(img, points_imu) 
            self.key_board.run(self.cam_intrinsic, self.cam_distort, cut_T_i2c, self.new_camera_intrinsic)
            cut_T_i2c = self.key_board._get_ext()
            print('------------rectify T:', cut_T_i2c)


def main():
    # config_path = '/root/cam2imu_manual_calib/src/cam2imu_manual_calib/cam2imu_manual_calib/config/new_GS4.yaml'
    config_path = '/root/cam2imu/config/GS4.yaml'
    cfg = ConfigLoader(config_path)

    cam2imu = Cam2imu(cfg)
    map_marks = cam2imu.map_server.get_points()
    cam2imu.run(map_marks)

if __name__ == "__main__":
    main()
