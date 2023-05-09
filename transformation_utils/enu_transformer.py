import numpy as np
import pyproj
import math
import scipy.spatial.transform
import pymap3d as pm
from .gnss_utils import *
from .ecef_transformer import ECEFTransformer

kSemimajorAxis = 6378137.0
kSemiminorAxis = 6356752.3142
kFirstEccentricitySquared = 0.00669437999014
kSecondEccentricitySquared = 0.00673949674228
kFlattening = 1.0 / 298.257223563
kEsq = kSemimajorAxis * kSemimajorAxis - kSemiminorAxis * kSemiminorAxis
kEarthRadius = 6378137.0

class ENUTransformer(object):
    def __init__(self):
        self.ecef_base_rot = np.identity(3)
        self.ecef_transformer = ECEFTransformer()
        self.base_map = {
            'chuansha': [31.206388889, 121.689444445]
        }

    # @staticmethod
    # def trans_geodetic2enu_use_pyproj(lat, lon, alt, lat_org, lon_org, alt_org, is_deg=True):
    #     transformer = pyproj.Transformer.from_crs(
    #         {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
    #         {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
    #         )

    #     x, y, z = transformer.transform(lon, lat, alt, radians=not is_deg)
    #     x_org, y_org, z_org = transformer.transform(lon_org, lat_org, alt_org,radians=not is_deg)
    #     vec=np.array([[ x-x_org, y-y_org, z-z_org]]).T

    #     rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=is_deg).as_matrix()#angle*-1 : left handed *-1
    #     rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=is_deg).as_matrix()#angle*-1 : left handed *-1

    #     rotMatrix = rot1.dot(rot3)

    #     enu1 = rotMatrix.dot(vec)
   
    #     enu = rotMatrix.dot(vec).T.ravel()
    #     return enu.T

    @staticmethod
    def trans_enu2geodetic_use_pyproj(x,y,z, lat_org, lon_org, alt_org, is_deg=True):
        transformer1 = pyproj.Transformer.from_crs(
            {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
            {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
            )
        transformer2 = pyproj.Transformer.from_crs(
            {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
            {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
            )
    
        x_org, y_org, z_org = transformer1.transform(lon_org, lat_org, alt_org, radians=not is_deg)
        ecef_org=np.array([[x_org,y_org,z_org]]).T
    
        rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=is_deg).as_matrix()#angle*-1 : left handed *-1
        rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=is_deg).as_matrix()#angle*-1 : left handed *-1

        rotMatrix = rot1.dot(rot3)

        ecefDelta = rotMatrix.T.dot( np.array([[x,y,z]]).T )
        ecef = ecefDelta+ecef_org
        lon, lat, alt = transformer2.transform(ecef[0,0], ecef[1,0], ecef[2,0],radians=not is_deg)
        return [lat,lon,alt]

    @staticmethod
    def trans_geodetic2enu_use_map3d(lat, lon, alt, lat_org, lon_org, alt_org, is_deg=True):
        return list(pm.geodetic2enu(lat, lon, alt, lat_org, lon_org, alt_org, deg=is_deg))

    def set_base(self, enu_base, is_deg=True, is_floor = True):
        self.enu_base = enu_base
        if not is_deg:
            self.enu_base[0] = np.rad2deg(enu_base[0])
            self.enu_base[1] = np.rad2deg(enu_base[1])
        self.geodetic_imu_matrix = self.ecef_transformer.get_geodetic2ecef(self.enu_base[0], self.enu_base[1],0,0,0,0)
        self.ecef_base_rot = self.geodetic_imu_matrix[0:3, 0:3]
        if is_floor:
            self.ecef_transformer.set_base(math.floor(self.geodetic_imu_matrix[0,3]),
                                            math.floor(self.geodetic_imu_matrix[1,3]),
                                            math.floor(self.geodetic_imu_matrix[2,3]))
        else:
            self.ecef_transformer.set_base(self.geodetic_imu_matrix[0,3],
                                            self.geodetic_imu_matrix[1,3],
                                            self.geodetic_imu_matrix[2,3])
    
    def get_ecef_trans(self):
        return self.ecef_transformer

    def get_geodetic2enu(self, lat, lon, alt, roll, pitch, yaw, is_deg=True):
        transformertest1 = pyproj.Transformer.from_crs(
            "epsg:4326", "epsg:32651"
            )
        x,y,z = transformertest1.transform(lat, lon, alt)
        enu_imu = np.array([x,y,z])
        pose = self.ecef_transformer.get_geodetic2global(lat, lon, alt, roll, pitch, yaw, is_deg)
        pose[0:3, 0:4] = np.dot(self.ecef_base_rot.T , pose[0:3, 0:4])
        pose[0,3] = enu_imu[0]
        pose[1,3] = enu_imu[1]
        pose[2,3] = enu_imu[2]
        return pose
    
    def trans_geodetic2enu(self, lat, lon, alt, is_deg=True):
        ecef_x, ecef_y, ecef_z = self.ecef_transformer.trans_geodetic2ecef(lat, lon, alt, is_deg)
        ecef_xyz = np.array([ecef_x, ecef_y, ecef_z])
        res = np.dot(self.ecef_base_rot.T, ecef_xyz)
        return res

    def trans_geodetic2enu_use_pyproj(self, lat, lon, alt, lat_org, lon_org, alt_org, is_deg=True):
        transformer = pyproj.Transformer.from_crs(
            {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
            {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
            )
        x, y, z = transformer.transform(lon, lat, alt, radians=not is_deg)
        x_org, y_org, z_org = transformer.transform(lon_org, lat_org, alt_org,radians=not is_deg)
        vec=np.array([[ x-x_org, y-y_org, z-z_org]]).T

        rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=is_deg).as_matrix()#angle*-1 : left handed *-1
        rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=is_deg).as_matrix()#angle*-1 : left handed *-1

        rotMatrix = rot1.dot(rot3)

        enu1 = rotMatrix.dot(vec)
   
        enu = rotMatrix.dot(vec).T.ravel()
        return enu.T

    def trans_enu2geodetic_use_pyproj(x,y,z, lat_org, lon_org, alt_org, is_deg=True):
        transformer1 = pyproj.Transformer.from_crs(
            {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
            {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
            )
        transformer2 = pyproj.Transformer.from_crs(
            {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
            {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
            )
    
        x_org, y_org, z_org = transformer1.transform(lon_org, lat_org, alt_org, radians=not is_deg)
        ecef_org=np.array([[x_org,y_org,z_org]]).T
    
        rot1 =  scipy.spatial.transform.Rotation.from_euler('x', -(90-lat_org), degrees=is_deg).as_matrix()#angle*-1 : left handed *-1
        rot3 =  scipy.spatial.transform.Rotation.from_euler('z', -(90+lon_org), degrees=is_deg).as_matrix()#angle*-1 : left handed *-1

        rotMatrix = rot1.dot(rot3)

        ecefDelta = rotMatrix.T.dot( np.array([[x,y,z]]).T )
        ecef = ecefDelta+ecef_org
        lon, lat, alt = transformer2.transform(ecef[0,0], ecef[1,0], ecef[2,0],radians=not is_deg)
        return [lat,lon,alt]

    def trans_geodetic2enu_use_map3d(lat, lon, alt, lat_org, lon_org, alt_org, is_deg=True):
        return list(pm.geodetic2enu(lat, lon, alt, lat_org, lon_org, alt_org, deg=is_deg))


if __name__ == '__main__':
    # The local coordinate origin (Zermatt, Switzerland)
    lat_org = 31.206388889 # deg
    lon_org = 121.689444445  # deg
    alt_org   = 1673     # meters

    roll = 0.355
    pitch = -0.149
    yaw = -342.265

    # The point of interest
    lat1 = 31.23369338 # deg
    lon = 121.66924814   # deg
    alt = 17.07      # meters
    
    transformertest1 = pyproj.Transformer.from_crs(
            "epsg:4326", "epsg:32651"
            )
    transformertest2 = pyproj.Transformer.from_crs(
            "epsg:32651","epsg:4326"
            )
    x_org, y_org, z_org = transformertest2.transform(lon_org, lat_org, alt_org)


    print(x_org, y_org, z_org)
    transfor = ENUTransformer()
    transfor.set_base([31.206388889, 121.689444445])
    #transfor.set_base([30.879573, 121.757103])

    # res1 = transfor.get_geodetic2enu(lat, lon, alt, roll, pitch, yaw)
    # res2 = transfor.trans_geodetic2enu(lat, lon, alt)
    res3 = transfor.trans_geodetic2enu_use_pyproj(lat1, lon, alt, lat_org, lon_org,alt_org)
    print(res3)
    # print(res2)