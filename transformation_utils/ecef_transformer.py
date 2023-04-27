import os
from .gnss_utils import *

class ECEFTransformer(object):
    def __init__(self):
        self.base_map = {
            'chuansha': [4396433, 598328, 4566559]
        }

    def set_base(self, base_coord_x, base_coord_y, base_coord_z):
        self.base_coord = [base_coord_x, base_coord_y, base_coord_z]
    
    def get_base(self):
        return self.base_coord

    def get_geodetic2global(self, lat, lon, alt, roll, pitch, yaw, is_deg=True):
        res = self.get_geodetic2ecef(lat, lon, alt, roll, pitch, yaw, is_deg)
        res[0,3] -= self.base_coord[0] 
        res[1,3] -= self.base_coord[1] 
        res[2,3] -= self.base_coord[2]
        return res 

    def get_geodetic2ecef(self, lat, lon, alt, roll, pitch, yaw, is_deg=True):
        lat_rad = lat
        lon_rad = lon
        roll_rad = roll
        pitch_rad = pitch
        yaw_rad = yaw
        if is_deg:
            lat_rad = np.deg2rad(lat)
            lon_rad = np.deg2rad(lon)
            roll_rad = np.deg2rad(roll)
            pitch_rad = np.deg2rad(pitch)
            yaw_rad = np.deg2rad(yaw)
        rot_body_ecef = geodetic_to_ecef_rot(lat_rad, lon_rad, alt, roll_rad, pitch_rad, yaw_rad)
        x, y, z = geodetic_to_ecef(lat_rad, lon_rad, alt, [0,0,0])
        res = np.identity(4)
        res[0:3, 0:3] = rot_body_ecef
        res[0, 3] = x
        res[1, 3] = y
        res[2, 3] = z
        return res

    def trans_geodetic2ecef(self, lat, lon, alt, is_deg=True):
        lat_rad = lat
        lon_rad = lon
        if is_deg:
            lat_rad = np.deg2rad(lat)
            lon_rad = np.deg2rad(lon)
        x, y ,z = geodetic_to_ecef(lat_rad, lon_rad, alt, self.base_coord)
        return x,y,z




