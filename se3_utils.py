import numpy as np
import math


class SE3:

    @staticmethod
    def rodrigues(v):
        # implementation follows that given in GTSAM
        ax, ay, az = v[0], v[1], v[2]
        SS = [[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]]
        theta2 = np.dot(v, v)
        I_3x3 = np.eye(3)
        if theta2 < 0.00000000001:
            return I_3x3 + SS
        theta = theta2 ** 0.5
        K = SS / theta
        KK = np.matmul(K, K)
        I_3x3 = np.eye(3)
        sin_theta = math.sin(theta)
        s2 = math.sin(theta / 2.0)
        one_minus_cos = 2.0 * s2 * s2
        R = I_3x3 + sin_theta * K + one_minus_cos * KK
        return R

    @staticmethod
    def logmap(m):
        # implementation follows that given in GTSAM
        R11, R12, R13 = m[0][0], m[0][1], m[0][2]
        R21, R22, R23 = m[1][0], m[1][1], m[1][2]
        R31, R32, R33 = m[2][0], m[2][1], m[2][2]
        tr = m[0][0] + m[1][1] + m[2][2]
        oe10 = 0.0000000001
        M_PI = 3.1515926
        if abs(tr + 1.0) < oe10:
            if abs(R33 + 1.0) > oe10:
                return (M_PI / (2 + 2.0 * R33) ** 0.5) * np.array([R13, R23, 1.0 + R33])
            elif abs(R22 + 1.0) > oe10:
                return (M_PI / (2 + 2.0 * R22) ** 0.5) * np.array([R12, 1.0 + R22, R32])
            else:
                return (M_PI / (2 + 2.0 * R11) ** 0.5) * np.array([1.0 + R11, R21, R31])
        else:
            tr_3 = tr - 3.0
            mag = 0
            if tr_3 < -0.0000001:
                theta = math.acos((tr - 1.0) / 2.0)
                mag = theta / (2.0 * math.sin(theta))
            else:
                mag = 0.5 - tr_3 ** 2 / 12.0
            return mag * np.array([R32 - R23, R13 - R31, R21 - R12])

    @staticmethod
    def pose_components(a):
        return a[0:3, 0:3], a[0:3, 3]

    @staticmethod
    def components_pose(r, t):
        mat = np.zeros((4, 4))
        mat[0:3, 0:3] = r
        mat[0:3, 3] = t
        mat[3, 3] = 1
        return mat

    @staticmethod
    def component_error(a, b):
        difference = SE3.between(a, b)
        return SE3.vector(difference)

    @staticmethod
    def pose_error(error):
        # error = SE3.between(a,b)
        r, t = SE3.pose_components(error)
        vecr = SE3.logmap(r)
        angle_error = np.linalg.norm(vecr)
        trans_error = np.linalg.norm(t)
        if math.isnan(trans_error):
            trans_error = 0
        if math.isnan(angle_error):
            angle_error = 0
        return angle_error, trans_error

    @staticmethod
    def get_pose(posevec):
        R = SE3.rodrigues(posevec[3:6])
        t = posevec[0:3]
        return SE3.components_pose(R, t)

    @staticmethod
    def vector(pose):
        R, t = SE3.pose_components(pose)
        flatr = SE3.logmap(R)
        return np.hstack([t, flatr])

    @staticmethod
    def identity():
        I = [0, 0, 0, 0, 0, 0]
        return SE3.get_pose(I)

    @staticmethod
    def invert(a):
        r, t = SE3.pose_components(a)
        return SE3.components_pose(r.transpose(), np.matmul(r.transpose(), -t))

    @staticmethod
    def compose(a, b):
        ar, at = SE3.pose_components(a)
        br, bt = SE3.pose_components(b)
        combr = np.matmul(ar, br)
        addt = np.matmul(ar, bt) + at
        return SE3.components_pose(combr, addt)

    @staticmethod
    def between(a, b):
        return SE3.compose(SE3.invert(a), b)

    @staticmethod
    def equals(a, b, tol=1e-4):
        btwn = SE3.between(a, b)
        btwn_vec = SE3.vector(btwn)
        for x in btwn_vec:
            if abs(x) > tol:
                return False
        return True


def test_identity():
    identity = SE3.identity()
    assert (np.sum(identity) == 4)


def test_invert():
    tR = [0.17299792, 2.61600057, 1.79300229, -
          1.63439723, -0.00351602, 0.02851071]
    mat = SE3.get_pose(tR)
    imat = SE3.invert(mat)
    res = SE3.compose(mat, imat)
    pose_error = SE3.pose_error(res, SE3.identity())
    for x in pose_error:
        assert (abs(x) < 0.00001)


def test_between():
    tR = [0.17299792, 2.61600057, 1.79300229, -
          1.63439723, -0.00351602, 0.02851071]
    mat = SE3.get_pose(tR)
    btwn = SE3.between(mat, mat)
    btwn = SE3.vector(btwn)
    for x in btwn:
        assert (abs(x) < 0.00001)


def test_norm():
    tR = [0.17299792, 2.61600057, 1.79300229, -
          1.63439723, -0.00351602, 0.02851071]
    pose = SE3.get_pose(tR)
    R, t = SE3.pose_components(pose)
    print('norm t: ', np.linalg.norm(t))
    newt = np.matmul(R.transpose(), t)
    print('norm newt: ', np.linalg.norm(newt))


if __name__ == '__main__':
    # test_identity()
    # test_invert()
    # test_between()
    test_norm()