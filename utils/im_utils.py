import cv2 as cv
import numpy as np


def angle2RT(angles):
    assert len(angles) == 3
    pitch, yaw, roll = angles  # orientation is a mess. Don't take the names srsly
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0.],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    R = Rz.dot(Ry.dot(Rx))
    RT = np.zeros((4, 4))
    RT[:3, :3] = R
    RT[-1, -1] = 1
    return RT


def proj_imu2cam(imu_pts, extrinsic, max_depth=20, min_depth=.5):
    """
    Project points from IMU coords into Cam coords
    :param imu_pts: (N, 3)
    :param extrinsic:
    :param max_depth:
    :param min_depth:
    :return: (N, 3)
    """
    cam_pts = np.dot(extrinsic[:3, :3], imu_pts.T).T + extrinsic[:3, 3]
    valid_idx = np.logical_and(cam_pts[:, 2] > min_depth, cam_pts[:, 2] < max_depth)
    cam_pts = cam_pts[valid_idx]
    return cam_pts


def proj_cam2img(cam_pts, intrinsic, distortion, new_camera_intrinsic, width=1024, height=576, cam_type='fisheye'):
    """
    Project points from Cam coords into img coords (pixels)
    """
    # FIXME: cannot associate input with output
    if cam_type == 'fisheye':
        # img_pts, _ = cv.fisheye.projectPoints(cam_pts.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)),
        #                                       intrinsic, distortion[:4])
        get_valid = True
        img_pts = project3DPtsToDSImg(cam_pts, intrinsic, get_valid)
    else:
        img_pts, _ = cv.projectPoints(cam_pts.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)),
                                      intrinsic, distortion)
    img_pts = img_pts[:, 0, :]
    valid_idx = (img_pts[:, 0] > 0) & (img_pts[:, 0] < width) & (img_pts[:, 1] > 0) & (img_pts[:, 1] < height)
    img_pts = img_pts[valid_idx]
    return img_pts

def proj_new_cam2img(cam_pts, camera_model, new_camera_intrinsic):
    pts_size = cam_pts.shape[0]
    img_pts = []
    for i in range(pts_size):
        img_pt = camera_model.Project(cam_pts[i])
        if camera_model.IsInCalibratedArea(img_pt[0]+3.5, img_pt[1]+3.5):
            img_pts.append(img_pt)
    img_pts = np.array(img_pts)
    valid_idx = (img_pts[:, 0] > camera_model.calibration_min_x()) & (img_pts[:, 0] < camera_model.calibration_max_x()) & (img_pts[:, 1] > camera_model.calibration_min_y()) & (img_pts[:, 1] < camera_model.calibration_max_y())
    img_pts = img_pts[valid_idx]

    return np.array(img_pts)

def pad_image(im, paddings, default_value=0):
    margin_paddings = np.array(paddings, dtype=np.int16) + 1
    h, w, c = im.shape
    ret = np.zeros((h + margin_paddings[1] + margin_paddings[3], w + margin_paddings[0] + margin_paddings[2], c), dtype=im.dtype)
    ret[:] = default_value
    ret[margin_paddings[1]:-margin_paddings[3], margin_paddings[0]:-margin_paddings[2]] = im
    ret = ret[1:-1, 1:-1]
    return ret


def coord_transform(H, src):
    """
    Given source points, calculate dst points after matrix transformation
    :param H: transformation matrix, shape (3x3)
    :param src: points, shape (N, 2)
    :return: dst points, shape (N, 2)
    """
    homo_src = np.vstack((src, np.ones(src.shape[1])))
    homo_dst = H.dot(homo_src)
    dst = np.array(homo_dst[:2] / (homo_dst[2] + 1e-9))
    return dst


def cal_map_from_homo(H, dst_size):
    """
    Calculate transformation map from Homography matrix
    :param H: homography matrix, shape (3x3)
    :param dst_size: target map size
    :return: target map
    """
    dst_indices = np.dstack(np.meshgrid(np.arange(dst_size[1]), np.arange(dst_size[0])))
    dst_vec = dst_indices.reshape((-1, 2)).transpose(1, 0)
    src_vec = coord_transform(H, dst_vec)
    mp = np.float32(src_vec.transpose(1, 0).reshape((dst_size[0], dst_size[1], 2)))
    return mp


def cal_map_from_homo_naive(H, dst_size):
    """
    Super slow. Only for verification use
    :param H:
    :param dst_size:
    :return:
    """
    mp = np.zeros((dst_size + (2,)))
    for y in xrange(dst_size[0]):
        for x in xrange(dst_size[1]):
            src_vec = coord_transform(H, np.array([x, y]).reshape(-1, 1))
            mp[y, x, 0] = src_vec[0]
            mp[y, x, 1] = src_vec[1]
    return np.float32(mp)


def is_pt_inside_grid(pts, grid):
    """
    Given points, judge if they are within the grid
    Not exact if grid is not rectangle.
    :param pts: (x, y), shape (N, 2), N = # of pts
    :param grid: [p1, p2, p3, p4], shape (4, 2)
        p1-------p2
        |        |
        p3-------p4
    :return: shape: (N,)
    """
    v1 = grid[1] - grid[0]
    v2 = grid[2] - grid[0]
    p = pts - grid[0]
    c1 = np.logical_and(0 <= np.inner(p, v1),  np.inner(p, v1) <= np.inner(v1, v1))
    c2 = np.logical_and(0 <= np.inner(p, v2),  np.inner(p, v2) <= np.inner(v2, v2))
    ret = np.logical_and(c1, c2)
    return ret


def cal_inv_map(src_mp, dst_size):
    """
    Calculate inverse map
    :param src_mp: OpenCV map, shape (im_h, im_w, 2), dtype np.float32 or CV_32FC2
    :param dst_size: inverse map size
    :return: inverse map
    """
    ret_mp = -np.ones(dst_size + (2,), dtype=np.float32)
    for y in xrange(src_mp.shape[0] - 1):
        for x in xrange(src_mp.shape[1] - 1):
            src_grid = np.array([(x, y), (x+1, y), (x, y+1), (x+1, y+1)], dtype=np.float32)
            dst_grid = src_mp[(y, y, y+1, y+1), (x, x+1, x, x+1)]
            xmax = min(dst_size[1], int(np.floor(max(dst_grid[:, 0]))) + 1)
            xmin = max(0, int(np.floor(min(dst_grid[:, 0]))))
            ymax = min(dst_size[0], int(np.floor(max(dst_grid[:, 1]))) + 1)
            ymin = max(0, int(np.floor(min(dst_grid[:, 1]))))
            if (xmax - xmin) > 100 or (ymax - ymin) > 100:  # Extrapolation problems
                continue
            H = cv.getPerspectiveTransform(dst_grid, src_grid)
            # TODO: vectorize?
            for yy in xrange(ymin, ymax):
                for xx in xrange(xmin, xmax):
                    if is_pt_inside_grid(np.array([xx, yy]), dst_grid):
                        ret_mp[yy, xx] = coord_transform(H, np.array([xx, yy]).reshape((-1, 1))).reshape(-1)
    # Fill the interpolation holes
    ret_mp[:, :, 0] = cv.morphologyEx(ret_mp[:, :, 0], cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    ret_mp[:, :, 1] = cv.morphologyEx(ret_mp[:, :, 1], cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return ret_mp


def map_homo_composition(mp, H, dst_size, mode=1):
    """
    Composite homography and transformation map
    :param mp: OpenCV map, shape (im_h, im_w, 2), dtype np.float32 or CV_32FC2
    :param H: Homography matrix, shape (3x3)
    :param dst_size: result map size
    :param mode:
        1: composition, ret_mp(im) = H(mp(im))
        2: inverse composition, im = ret_mp(H(mp(im))
    :return: ret_mp: result map
    """
    ret_mp = -np.ones(dst_size + (2,), dtype=np.float32)
    inter_grids = np.array([[[(x, y), (x+1, y), (x, y+1), (x+1, y+1)]
                             for x in xrange(mp.shape[1]-1)] for y in xrange(mp.shape[0]-1)])
    src_grids = np.array([[mp[(y, y, y+1, y+1), (x, x+1, x, x+1)]
                           for x in xrange(mp.shape[1]-1)] for y in xrange(mp.shape[0]-1)], dtype=np.float32)
    dst_pts = coord_transform(H, inter_grids.reshape((-1, 2)).transpose()).astype(np.float32)
    dst_grids = dst_pts.transpose().reshape((mp.shape[0]-1, mp.shape[1]-1, 4, 2))
    if mode != 1:
        dst_grids, src_grids = src_grids, dst_grids
    for y in xrange(mp.shape[0] - 1):
        for x in xrange(mp.shape[1] - 1):
            src_grid = src_grids[y, x]
            dst_grid = dst_grids[y, x]
            xmax = min(dst_size[1], int(np.floor(max(dst_grid[:, 0]))) + 1)
            xmin = max(0, int(np.floor(min(dst_grid[:, 0]))))
            ymax = min(dst_size[0], int(np.floor(max(dst_grid[:, 1]))) + 1)
            ymin = max(0, int(np.floor(min(dst_grid[:, 1]))))
            if (xmax - xmin) > 100 or (ymax - ymin) > 100:  # Extrapolation problems
                continue
            grid_H = cv.getPerspectiveTransform(dst_grid, src_grid)
            # TODO: vectorize?
            for yy in xrange(ymin, ymax):
                for xx in xrange(xmin, xmax):
                    if is_pt_inside_grid(np.array([xx, yy]), dst_grid):
                        ret_mp[yy, xx] = coord_transform(grid_H, np.array([xx, yy]).reshape((-1, 1))).reshape(-1)

    # Fill the interpolation holes
    ret_mp[:, :, 0] = cv.morphologyEx(ret_mp[:, :, 0], cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    ret_mp[:, :, 1] = cv.morphologyEx(ret_mp[:, :, 1], cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return ret_mp

def dscamera2pixel( p_c, param):
    fx = param[0]
    fy = param[1]
    cx = param[2]
    cy = param[3]

    xi = param[4]
    alpha = param[5]
    
    x = p_c[0]
    y = p_c[1]
    z = p_c[2]
    
    xx = x * x
    yy = y * y
    zz = z * z

    r2 = xx + yy
    d1_2 = r2 + zz
    d1 = np.sqrt(d1_2)
    w1 = (1.0 - alpha) / alpha if alpha > 0.5 else alpha / (1.0 - alpha)
    w2 = (w1 + xi) / np.sqrt(2.0 * w1 * xi + xi * xi + 1.0)
    valid = bool(z > -w2 * d1)

    k = xi * d1 + z
    kk = k * k

    d2_2 = r2 + kk
    d2 = np.sqrt(d2_2)
    norm = alpha * d2 + (1.0 - alpha) * k

    mx = x / norm
    my = y / norm
    proj = (0,0)
    proj[0] = fx * mx + cx
    proj[1] = fy * my + cy 

    return proj, valid

def pixel2dscamera(proj, param, valid):
    fx = param[0]
    fy = param[1]
    cx = param[2]
    cy = param[3]

    xi = param[4]
    alpha = param[5]

    mx = (proj[0] - cx) / fx
    my = (proj[1] - cy) / fy

    r2 = mx * mx + my * my
    valid = not(alpha > 0.5) and ((r2 >= 1.0) / 2.0 * alpha - 1.0)
    
    xi2_2 = alpha * alpha
    xi1_2 = xi * xi

    sqrt2 = np.sqrt(1.0 - (2 * alpha - 1.0) * r2)
    norm2 = alpha * sqrt2  + 1.0 - alpha

    mz = (1.0 - xi2_2 * r2) / norm2
    mz2 = mz * mz
    norm1 = mz2 + r2
    sqrt1 = np.sqrt(mz2 + (1.0 - xi1_2)* r2)
    k = (mz * xi + sqrt1) / norm1
    p3d = np.array([0,0,0])
    p3d[0] = k * mx
    p3d[1] = k * my
    p3d[2] = k * mz - xi
    
    return p3d

def project3DPtsToDSImg(pts_3d, param, get_valid):
    pts_2d = []
    pts_size = np.size(pts_3d)
    for i in range(pts_size):
        tmp_point = pts_3d[i]
        # R = tfrom[0:3, 0:3]
        # t = tfrom[0:3, 3]
        # pt_tfrom = R * tmp_point + t
        pt_piexl, valid= dscamera2pixel(tmp_point, param)
        if get_valid:
            if valid:
                pts_2d.append(pt_piexl)
        else:
            pts_2d.append(pt_piexl)

    return pts_2d


