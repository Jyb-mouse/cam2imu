'''
Author: jin yanbin
email: jinyanbin@maxieyetech.com
Date: 2023-02-13 16:06:33
LastEditTime: 2023-04-21 10:47:10
filePath: Do not edit
'''

from bag_reader import RosbagReader
import yaml
import numpy as np

class ConfigLoader(object):
    def __init__(self, config_path):
        self.cfg = self.load_cfg(config_path)
        self.geojson_path = self.cfg['geojson_path']
        self.bag_path = self.cfg['bag_file']
        self.output_path = config_path
        self.ts_begin = self.cfg['ts_begin']
        self.ts_end = self.cfg['ts_end']
        self.base = self.cfg['base']
        self.topics = self.cfg['topics']
        self.aligner = self.cfg['aligner']
        self.cam_id = self.cfg['cam_id']
        self.imu_id = self.cfg['imu_id']
        self.type = self.cfg['type']
        self.width = self.cfg['width']
        self.height = self.cfg['height']
        self.cam_intrinsic = np.array(self.cfg['cam_intrinsic'])
        self.cam_distort = np.array(self.cfg['cam_distort'])
        self.extrinsic = np.array(self.cfg['extrinsic'])
        self.extrinsic_ori = np.array(self.cfg['extrinsic_ori'])
        self.bag_reader = RosbagReader(self.bag_path, self.base)
        self.bag_reader.init_topics(self.aligner, self.topics)

    def load_cfg(self, cfg_path):
        with open(cfg_path, 'r') as f:
            config_file = yaml.safe_load(f)
        return config_file