import numpy as np
import cv2 as cv
import yaml

# a = (1.154139, -1.136627, 1.241442)
a = (1.186983, -1.181562, 1.204199)

R = cv.Rodrigues(a)

print(R[0])

R_l2c = R[0]
t_l2c = np.array([-0.148348, 1.516199, 1.456540])
T_l2c = np.identity(4)
T_l2c[:3,:3] = R_l2c
T_l2c[:3, 3] = t_l2c

print(T_l2c)
print(np.linalg.inv(T_l2c))

# T_l2i = [[ 0.99997836, -0.00581856, -0.00307055,  0.0561366 ],
#  [ 0.00581448,  0.9999822 , -0.00133604, -0.00265236],
#  [ 0.00307827,  0.00131816,  0.99999439,  0.14312953],
#  [ 0.        ,  0.        ,  0.        ,  1.        ]]

# # T_i2l = np.linalg.inv(T_l2i)

# # T_i2c = np.dot(T_i2l,T_l2c)
# # print(T_i2c)
# output_dir = './output/a.yaml'

# # T_imu2cam = np.linalg.inv(extrinsic)

# with open(output_dir,'r',encoding='utf-8') as f:
#     d = yaml.load(f.read(), Loader=yaml.FullLoader)
# print(d['extrinsic'])
# d['extrinsic'] = T_l2i

# with open(output_dir, 'w',encoding = 'utf-8') as f:
#     # yaml.dump(d, f, allow_unicode=True)
#     f.write(yaml.dump(d))
# f = open(output_dir, 'w')
# yaml.dump(doc, f)