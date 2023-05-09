import os
import math
import pygame
import cv2 as cv
import numpy as np
import yaml


from transformation_utils.transform_utils import Transmat2RT #NewCameraModel
from pprint import pprint

from utils import im_utils

np.set_printoptions(formatter={'float': lambda x: '{:.4f}'.format(x)})


class Background(pygame.sprite.Sprite):
    def __init__(self, location=(0, 0)):
        super(Background, self).__init__()
        self.location = location
        self.image = None
        self.rect = None

    def set_img(self, img_str, height, width):
        self.image = pygame.image.frombuffer(img_str, (height, width), 'BGR')
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = self.location


class Board(object):
    def __init__(self, output_dir, cam_id, imu_id, cam_type, width, height, **kwargs):
        self.output_dir = output_dir
        self.cam_id = cam_id
        self.imu_id = imu_id
        self.cam_type = cam_type
        self.img_shape = (width, height)
        self.width = width
        self.height = height
        self.max_depth = kwargs.get('max_depth', 300)
        self.min_depth = kwargs.get('min_depth', 0)
        self.bag_name = kwargs.get('bag_name')
        
        # set depth
        self.init_pts_depth()

        self.imu_anchors = []  # in IMU coords
        self.init_extrinsic = None
        self.ext_i2c = None
        self.img = None
        self.bg = Background()
        pygame.init()
        # os.environ["SDL_VIDEODRIVER"] = "dummy"
        self.screen = pygame.display.set_mode((self.width, self.height))

        # self.step_keys = {'1': 0.0001,
        #                   '2': 0.001,
        #                   '3': 0.01,
        #                   '4': 0.1,
        #                   '5': 1}
        # self.translation_keys = {'q': (1, 0, 0), 'a': (-1, 0, 0),
        #                          'w': (0, 1, 0), 's': (0, -1, 0),
        #                          'e': (0, 0, 1), 'd': (0, 0, -1)}
        # self.angle_keys = {'r': (1, 0, 0), 'f': (-1, 0, 0),
        #                    't': (0, 1, 0), 'g': (0, -1, 0),
        #                    'y': (0, 0, 1), 'h': (0, 0, -1)}
        # self.next_keys = {'m'}
        # self.save_keys = {'n'}
        # self.quit_keys = {'b'}
        # self.upload_keys = {'v'}
        self.step_keys = {pygame.K_1: 0.0001,
                          pygame.K_2: 0.001,
                          pygame.K_3: 0.01,
                          pygame.K_4: 0.1,
                          pygame.K_5: 1}
        self.translation_keys = {pygame.K_d: (1, 0, 0), pygame.K_a: (-1, 0, 0),
                                 pygame.K_s: (0, 1, 0), pygame.K_w: (0, -1, 0),
                                 pygame.K_z: (0, 0, 1), pygame.K_x: (0, 0, -1)}
        self.angle_keys = {pygame.K_UP: (1, 0, 0), pygame.K_DOWN: (-1, 0, 0),
                           pygame.K_LEFT: (0, 1, 0), pygame.K_RIGHT: (0, -1, 0),
                           pygame.K_n: (0, 0, 1), pygame.K_m: (0, 0, -1)}
        self.next_keys = {pygame.K_TAB}
        self.save_keys = {pygame.K_RETURN}
        self.quit_keys = {pygame.K_q}
        self.upload_keys = {pygame.K_u}

        pygame.display.set_caption("Camera Calibration")
        self.step = 1.0
        self.clock = pygame.time.Clock()


    def init_pts_depth(self):
        if self.cam_id in['3', '53']:
            self.min_depth = 15
        elif self.cam_id in ['4', '17', '54', '57']:
            self.min_depth = 25
        elif self.cam_id in['1', '2', '5', '8', '9', '12', '13', '51', '58', '59']:
            self.min_depth = 2
        elif self.cam_id in ['6', '7']:
            self.min_depth = 10
        elif self.cam_id in ['14', '15']:
            self.max_depth = 10
            self.min_depth = 2
        else:
            raise RuntimeError('No cam_id input!')
        print("max_depth: ", self.max_depth)
        print("min_depth: ", self.min_depth)

    def set_input(self, im, pts):
        im = cv.resize(im, self.img_shape)
        self.img = im
        im_str = im.tostring()

        self.bg.set_img(im_str, self.width, self.height)
        self.imu_anchors = pts

    def _get_ext(self):
        return self.ext_i2c

    def _reset(self):
        self.ext_i2c = self.init_extrinsic.copy()

    def _set_init_extrinsic(self, extrinsic):
        if self.init_extrinsic is None:
            self.init_extrinsic = extrinsic.copy()
            self._reset()

    def _rectify(self, intrinsic, distortion, new_camera_intrinsic):
        # get_valid = False
        # img_pts = im_utils.project3DPtsToDSImg(self.imu_anchors, intrinsic, get_valid, self.ext_i2c)
        cam_pts = im_utils.proj_imu2cam(
            self.imu_anchors, self.ext_i2c, max_depth=100, min_depth=3)
        if (cam_pts.shape[0] > 0):
            img_pts = im_utils.proj_cam2img(cam_pts, intrinsic, distortion, 
                             new_camera_intrinsic, self.width, self.height, cam_type=self.cam_type)
            lenth = len(img_pts)
            for i in range(0,lenth):
                pygame.draw.circle(self.screen, (255, 0, 0), (int(img_pts[i][0]), int(img_pts[i][1])), 3, 2)
                # cv.circle(self.img, (int(img_pts[i][0]), int(img_pts[i][1])), 5, (0, 140, 255), -1)
                if i < lenth / 2 :
                    if int(img_pts[i][0]) == int(img_pts[i+4][0]):
                        pygame.draw.line(self.screen, (0, 255, 0), (int(img_pts[i][0]),int(img_pts[i][1])), (int(img_pts[i+1][0]),int(img_pts[i+1][1])), 2)
                        pygame.draw.line(self.screen, (0, 255, 0), (int(img_pts[i+1][0]),int(img_pts[i+1][1])), (int(img_pts[i+2][0]),int(img_pts[i+2][1])),  2)
                        pygame.draw.line(self.screen, (0, 255, 0), (int(img_pts[i+2][0]),int(img_pts[i+2][1])), (int(img_pts[i+3][0]),int(img_pts[i+3][1])),  2)
                        pygame.draw.line(self.screen, (0, 255, 0), (int(img_pts[i+3][0]),int(img_pts[i+3][1])), (int(img_pts[i+4][0]),int(img_pts[i+4][1])),  2)
            # cv.imwrite('./old.jpg',self.img)
    def save_imu2cam_extrinsic(self, output_dir, extrinsic):
        T_imu2cam = extrinsic
        T_imu2cam = T_imu2cam.tolist()
        with open(output_dir,'r',encoding='utf-8') as f:
            d = yaml.load(f.read(), Loader=yaml.FullLoader)
        d['extrinsic'] = T_imu2cam

        with open(output_dir, 'w',encoding = 'utf-8') as f:
            # yaml.dump(d, f, allow_unicode=True)
            f.write(yaml.dump(d))

    def run(self, intrinsic, distortion, extrinsic, new_camera_intrinsic):
        """
        main loop
        :return: exit flag
        """
        self._set_init_extrinsic(extrinsic)

        angle_delta = np.zeros(3)
        trans_delta = np.zeros(3)
        while 1:
            self.screen.fill([255, 255, 255])
            self.screen.blit(self.bg.image, self.bg.rect)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key in self.next_keys:
                        return False
                    if event.key in self.save_keys:
                        self.save_imu2cam_extrinsic(self.output_dir, self._get_ext())
                        print ('imu2cam extrinsic saved!')
                    elif event.key in self.quit_keys:
                        pygame.quit()
                        return True
                    elif event.key in self.step_keys:
                        self.step = self.step_keys[event.key]
                    elif event.key in self.translation_keys:
                        trans_delta += np.array(self.translation_keys[event.key]) * self.step
                    elif event.key in self.angle_keys:
                        angle_delta += np.array(self.angle_keys[event.key]) * self.step * math.pi
                    elif event.key == pygame.K_SPACE:
                        self._reset()
                    print ('current step: ', 1.0 / self.step)

                # TODO: Pythonify these
                if event.type == pygame.KEYUP:
                    # stop rotating
                    if event.key in {pygame.K_UP, pygame.K_DOWN}:
                        angle_delta[0] = 0
                    if event.key in {pygame.K_LEFT, pygame.K_RIGHT}:
                        angle_delta[1] = 0
                    if event.key in {pygame.K_n, pygame.K_m}:
                        angle_delta[2] = 0
                    # stop translate
                    if event.key in {pygame.K_a, pygame.K_d}:
                        trans_delta[0] = 0
                    if event.key in {pygame.K_w, pygame.K_s}:
                        trans_delta[1] = 0
                    if event.key in {pygame.K_z, pygame.K_x}:
                        trans_delta[2] = 0
                    if event.key == pygame.K_p:
                        # pprint(np.linalg.inv(self.ext_i2c))
                        pprint(self.ext_i2c)
                        pprint(np.rad2deg(Transmat2RT(self._get_ext())[0]))

            self._update_ext(angle_delta, trans_delta)
            self.clock.tick(100)
            self._rectify(intrinsic, distortion, new_camera_intrinsic)
            pygame.display.flip()

    def _update_ext(self, angles, translations):
        R = im_utils.angle2RT(angles)
        self.ext_i2c = R.dot(self.ext_i2c)
        _trans = translations.reshape(3)
        self.ext_i2c[:3, -1] += _trans


def main():
    img = cv.imread('/home/mouse/Documents/cam-imu/image/1668372322.076.jpg')
    output_dir = "/home/mouse/Documents/cam2imu_manual_calib/src/cam2imu_manual_calib/cam2imu_manual_calib/output/"
    cam_id = '3'
    imu_id = 0
    cam_type = 'pinhole'
    window_size = (1824, 944)
    points = [[-1.2278, 0.0525, 54.7411],
                [-1.1541, -0.0641, 60.7585],
                [-1.3040, -0.0698, 60.7602]]
    points = np.array(points)
    key_board = Board(output_dir, cam_id, imu_id, cam_type, window_size)
    key_board.set_input(img, points)
    cam_intrinsic = np.array([
            [1464.027239, 0.0, 927.68845679],
            [0.0, 1463.56382896, 469.74350957],
            [0.0, 0.0, 1.0]
        ])
    cam_distort = np.array(
        [5.71331, 1.6418, -0.0000979, -0.0000992, -0.130419, 7.0173, 6.01327, 0.0]
    )
    T_i2c =np.array([[  0.99991425,  0.01196087,  0.0053313 , -0.01838986],
            [ 0.00537511, -0.00363349, -0.99997895, -0.51552734],
            [-0.01194125,  0.99992186, -0.00369747, -1.18418307],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    new_camera_intrinsic = np.array([
            [1464.027239, 0.0, 927.68845679],
            [0.0, 1463.56382896, 469.74350957],
            [0.0, 0.0, 1.0]
        ])
    key_board.run(cam_intrinsic, cam_distort, T_i2c, new_camera_intrinsic)


if __name__ == "__main__":
    main()