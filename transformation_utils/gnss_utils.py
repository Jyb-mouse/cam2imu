import math
import numpy as np

kSemimajorAxis = 6378137.0
kSemiminorAxis = 6356752.3142
kFirstEccentricitySquared = 0.00669437999014
kSecondEccentricitySquared = 0.00673949674228
kFlattening = 1.0 / 298.257223563
kEsq = kSemimajorAxis * kSemimajorAxis - kSemiminorAxis * kSemiminorAxis
kEarthRadius = 6378137.0

def enu_to_ned_rot():
    res = np.matrix([[0.0,1.0,0.0],
              [1.0,0.0,0.0],
              [0.0,0.0,-1.0]])
    return res

def ecef_to_ned_rot(lat_rad, lon_rad):
    cos_lat = math.cos(lat_rad)
    sin_lat = math.sin(lat_rad)
    cos_lon = math.cos(lon_rad)
    sin_lon = math.sin(lon_rad)

    ecef_ned = np.matrix([
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [-sin_lon, cos_lon, 0.0],
        [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
    ])
    return ecef_ned

def geodetic_to_ecef(lat, lon, alt, base_coord):
    xi = math.sqrt(1 - kFirstEccentricitySquared * math.sin(lat) * math.sin(lat))
    res_x = (kSemimajorAxis / xi + alt) * math.cos(lat) * math.cos(lon) - base_coord[0]
    res_y = (kSemimajorAxis / xi + alt) * math.cos(lat) * math.sin(lon) - base_coord[1]
    res_z = (kSemiminorAxis / kSemimajorAxis * kSemiminorAxis / xi + alt) * math.sin(lat) - base_coord[2]
    return res_x, res_y, res_z

def geodetic_to_ecef_rot(lat, lon, alt, roll, pitch, yaw):
    rot_body_enu = geodetic_to_enu_rot(roll, pitch, yaw)
    rot_enu_ned = enu_to_ned_rot()
    rot_ecef_ned = ecef_to_ned_rot(lat, lon)
    rot_body_ecef = rot_ecef_ned.T * rot_enu_ned * rot_body_enu
    return rot_body_ecef

def geodetic_to_enu_implementation(lat_deg, lon_deg, enu_base):
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    base_lat_rad = np.deg2rad(enu_base[0])
    base_lon_rad = np.deg2rad(enu_base[1])

    coslat_rad = math.cos(lat_rad)
    coslon_rad = math.cos(lon_rad)
    sinlat_rad = math.sin(lat_rad)
    sinlon_rad = math.sin(lon_rad)
    xx = coslat_rad * coslon_rad * math.cos(base_lon_rad) * math.cos(base_lat_rad) \
        + coslat_rad * sinlon_rad * math.sin(base_lon_rad) * math.cos(base_lat_rad) \
        + sinlat_rad * math.sin(base_lat_rad)
    yy = -coslat_rad * coslon_rad * math.sin(base_lon_rad) + coslat_rad * sinlon_rad * math.cos(base_lon_rad)
    zz = -coslat_rad * coslon_rad * math.cos(base_lon_rad) * math.sin(base_lat_rad) \
        - coslat_rad * sinlon_rad * math.sin(base_lon_rad) * math.sin(base_lat_rad) \
        + sinlat_rad * math.cos(base_lat_rad)
    x = math.atan2(yy, xx) * kEarthRadius
    y = math.log(math.tan(math.asin(zz) / 2 + math.pi / 4)) * kEarthRadius
    return x, y

def geodetic_to_enu_rot(roll, pitch, yaw):
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