import os
import sys
import time
import math
import argparse
import numpy as np
import cv2 as cv
import signal

# import bagpy
# from bagpy import bagreader
import pandas as pd

import rospy
import rosbag
from cv_bridge import CvBridge
# from sensor_msgs.msg import Image

from scipy.spatial.transform import Rotation as R


# make numpy print prettier
np.set_printoptions(suppress=True)

def signal_exit(signal, frame):
    sys.exit(1)

#helper to constrain certain arguments to be specified only once
class Once(argparse.Action):
    def __call__(self, parser, namespace, values, option_string = None):
        if getattr(namespace, self.dest) is not None:
            msg = '{o} can only be specified once'.format(o = option_string)
            raise argparse.ArgumentError(None, msg)
        setattr(namespace, self.dest, values)

def parseArgs():
    class KalibrArgParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_help()
            sm.logError('%s' % message)
            sys.exit(2)
        def format_help(self):
            formatter = self._get_formatter()
            formatter.add_text(self.description)
            formatter.add_usage(self.usage, self._actions,
                                self._mutually_exclusive_groups)
            for action_group in self._action_groups:
                formatter.start_section(action_group.title)
                formatter.add_text(action_group.description)
                formatter.add_arguments(action_group._group_actions)
                formatter.end_section()
            formatter.add_text(self.epilog)
            return formatter.format_help()

    usage = """ 
    Example usage:
    python %(prog)s --rosbag 2019-12-18-14-37-11.bag --begin 0 --end -1 --imu_topic /met/imu  --ori_topic /met/gps/orientation --out_path ../../../
    """

    #setup the argument list
    parser = KalibrArgParser(description="Convert rosbag",usage=usage)

    #rosbag source
    groupData = parser.add_argument_group("Rosbag")
    groupData.add_argument('--rosbag',dest='rosbag_name',nargs=1,help='rosbag name containing gps and imu data',action=Once, required=True)
    groupData.add_argument('--begin',dest='begin',nargs=1,help ="begin timestamp",action=Once, required=True)
    groupData.add_argument('--end',dest='end',nargs=1,help="end timestamp",action=Once, required=True)

    #configuration files
    groupTopic = parser.add_argument_group("Topic")
    groupTopic.add_argument('--imu_topic',nargs='+',dest='imu_topic',help='camimuera topic in rosbag',action=Once, required=True)
    groupTopic.add_argument('--ori_topic',nargs='+',dest='ori_topic',help='gps orientation topics in rosbag',action=Once, required=True)

    #output
    groupOutput = parser.add_argument_group("Output")
    groupOutput.add_argument("--out_path",dest='out_path',nargs=1,help='output rosbag path')

    #print help if no argument is specified
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(2)

    #Parser the argument list
    try:
        parsed = parser.parse_args()
    except:
        sys.exit(2)

    return parsed

class DatasetConverter(object):
    def __init__(self,rosbag_path, img_path, img_topic, out_path):
        self.rosbag_path = rosbag_path
        self.rosbag_path_new = self.insert_str(rosbag_path, '_new')
        self.rosbag_path_merge = self.insert_str(rosbag_path, '_merge1')
        self.img_path = img_path
        self.img_topic = img_topic
        self.cv_brige = CvBridge()
        self.out_path = os.path.join(os.path.dirname(__file__),
            out_path)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        try:
            self.img_list = []
            self.get_image_list(self.img_path, 'jpg')
            self.img_list.sort()
            print(len(self.img_list))
            self.bag_new = rosbag.Bag(self.rosbag_path_new,'w',compression = rosbag.Compression.NONE)
        except Exception as e:
            print("open file fail!")
            raise Exception(e)

    def get_newbag_name(self):
        return self.rosbag_path_new

    def get_mergebag_name(self):
        return self.rosbag_path_merge

    def get_image_list(self, dir_path, ext=None):
        newDir = dir_path
        if os.path.isfile(dir_path):
            if ext is None:
                self.img_list.append(dir_path)
            else:
                if ext in dir_path[-3:]:
                    self.img_list.append(dir_path)

        elif os.path.isdir(dir_path):
            for s in os.listdir(dir_path):
                newDir=os.path.join(dir_path,s)
                self.get_image_list(newDir, ext)

    def get_img_timestamp(self, img_name):
        split_str = img_name.split('/')[-1]
        index = split_str.split('.')
        if len(index) == 3:
            sec = float(index[0])
            nsec = float('.'+index[1])
            timestamp = sec+nsec
        elif len(index) == 2:
            timestamp = float(index[0])
        else:
            print("wrong img format!")
            timestamp = 0
        return rospy.rostime.Time.from_sec(timestamp)

    @staticmethod
    def insert_str(ori_str, in_str):
        str_list = list(ori_str)
        nPos = str_list.index('.')
        str_list.insert(nPos, in_str)
        res_str = "".join(str_list)
        return res_str

    def convert(self):
        print("converting...")
        for image_name in self.img_list:
            stamp = self.get_img_timestamp(image_name)
            im = cv.imread(image_name)
            if im.shape[0] > 0 and im.shape[1] > 0:
                image = self.cv_brige.cv2_to_compressed_imgmsg(im)
                image.header.stamp = stamp
                image.header.frame_id = 'camera'
                self.bag_new.write(self.img_topic, image, stamp)
            else:
                print("NO align data for {}".format(stamp))
        print("Convert finished!!!")
        self.bag_new.close()

def main():
    # Parse the arguments
    # parsed = parseArgs()

    # signal.signal(signal.SIGINT, signal_exit)
    # init converter
    # bag_converter = DatasetCo/met/gps/orientationnverter(parsed.dataset_name,parsed.cam_topics,parsed.imu_topics,parsed.out_path[0])
    rosbag_name = '/home/mouse/data_ws/bag/A12/1_20230413_055920_005.bag'
    img_path = '/home/mouse/data_ws/img/A12/img_ori'
    img_topic = '/camera1/image/compressed_new'
    out_path = '/home/mouse/Documents/cam-imu/new'


    bag_converter = DatasetConverter(rosbag_name,img_path,img_topic,out_path)
    #run 
    bag_converter.convert()
    rosbag_name_new = bag_converter.get_newbag_name()
    rosbag_name_merge = bag_converter.get_mergebag_name()


    # file_path = os.path.join(os.path.dirname(__file__), "bag_merge.py")
    # os.system("python3 {} {} {} -o {}".format(file_path, rosbag_name, rosbag_name_new, rosbag_name_merge))


if __name__ == "__main__":
    main()