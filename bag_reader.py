import yaml
import numpy as np
import rosbag
from rospy.rostime import Time
from copy import deepcopy
from se3_utils import SE3
# from py_lidar import trans_pointcloud2_to_array

from aligner import align
from transformation_utils import ENUTransformer


class MsgReader(object):
    def __init__(self, bag, name, topic):
        self.bag = bag
        self.name = name
        self.topic = topic
        self.meta = self.bag.read_messages('{}'.format(topic))
    
    def get_rows(self, ts_begin, ts_end):
        start = Time(ts_begin/1e9)
        end = Time(ts_end/1e9)
        self.meta = self.bag.read_messages('{}'.format(self.topic), start_time=start, end_time=end)
        for topic,msg,time in self.meta:
            ts = msg.header.stamp.to_nsec()
            obj = deepcopy(msg)
            yield ts, obj


class RosbagReader(object):
    def __init__(self, bag_path, base_point=None):
        self.tpoic_keys, self.topics = None,None
        self.bag_path = bag_path
        self.base_point = base_point
        self.meta_list = []
        self.enu_transformer = ENUTransformer()
        self.enu_transformer.set_base(base_point)
        self.ecef_transformer = self.enu_transformer.get_ecef_trans()

        try:
            self.bag = rosbag.Bag(self.bag_path, "r")
            self.info_dict = yaml.safe_load(rosbag.Bag(self.bag_path, 'r')._get_yaml_info())
            self.bag_start = self.info_dict['start']*1e9
            self.bag_end = self.info_dict['end']*1e9
            self.bag_duration = self.info_dict['duration']
        except Exception as e:
            print("open bag: {} fail!".format(self.bag_path))
            raise Exception(e)

    def process_ts(self, ts):
        if ts is None:
            return None
        if isinstance(ts, float):
            return int(ts)
        if isinstance(ts, int):
            return ts
        if isinstance(ts, str):
            base = self.bag_start
            multiplier = 1
            if ts[0] == '-':
                ts = ts[1:]
                base = self.bag_end
                multiplier = -1
            amount = 0
            for seg in ts.split(':'):
                amount = amount * 60 + float(seg)
            return base + multiplier * int(round(amount*1e9))
        else:
            raise TypeError('Invalid ts type {}'.format(type(ts)))

    def init_topics(self, aligner, topics_desp):
        self.aligner_topic = aligner
        self.topic_keys, self.topics = [], []
        self.topics_desp = topics_desp

        for key, desp in topics_desp.items():
            if type(desp) is str:
                self.topic_keys += [key]
                self.topics += [desp]
                self.meta_list += [MsgReader(self.bag, key, desp)]
            elif type(desp) is dict:
                for item_key, item_desp in desp.items():
                    self.topic_keys += ['{}{}'.format(key, item_key)]
                    self.topics += ['{}'.format(item_desp)]
        self.topic_keys_total = ['aligner']+self.topic_keys
        print(self.topic_keys_total)

    def geodetic_msg_to_loc_ecef(self, gps):
        try:
            assert gps is not None
        except AssertionError:
            print('[map_service] Error: NO GPS data')
            return None

        position = np.array([gps.latitude, gps.longitude, gps.altitude])

        if np.isnan(position[-1]):
            raise RuntimeError('{} not in the map'.format(position))

        # roll pitch yaw
        orientation = [gps.position_covariance[1], gps.position_covariance[2], -gps.position_covariance[3]]
        loc = np.array([position, orientation])
        # self.ecef_transformer.set_base(4396433, 598328, 4566559)
        ins_dict = self.ecef_transformer.get_geodetic2global(position[0], position[1], position[2], 
                                                           orientation[0], orientation[1], orientation[2], is_deg=True)

        return ins_dict

    def geodetic_msg_to_loc_enu(self, gps):
        try:
            assert gps is not None
        except AssertionError:
            print('[map_service] Error: NO GPS data')
            return None

        position = np.array([gps.latitude, gps.longitude, gps.altitude])

        if np.isnan(position[-1]):
            raise RuntimeError('{} not in the map'.format(position))

        # roll pitch yaw
        orientation = [gps.position_covariance[1], gps.position_covariance[2], -gps.position_covariance[3]]
        loc = np.array([position, orientation])
        ins_dict = self.enu_transformer.get_geodetic2enu(position[0], position[1], position[2], 
                                                           orientation[0], orientation[1], orientation[2], is_deg=True)
        # utm
        # ins_dict = self.enu_transformer.get_geodetic2enu_use_pyproj(position[0], position[1], position[2], 
        #                                                    orientation[0], orientation[1], orientation[2], is_deg=True)

        return ins_dict

    def fetch_near(self, topic, ts, hz):
        ts_begin = self.process_ts(self.process_ts(ts) - (1.0/hz*1e9))
        ts_end = self.process_ts(self.process_ts(ts) + (1.0/hz*1e9))
        msg_reader = MsgReader(self.bag, 'topic', topic)
        meta = msg_reader.get_rows(ts_begin, ts_end)

        diff = 10**9
        res_t, res_data = None, None
        for t, data in meta:
            if abs(ts - t) < diff:
                res_t = t
                res_data = data
                diff = abs(ts - t)
        return (res_t, res_data)

    def fectch_align(self, ts_begin, ts_end, limit):
        options = {
            'ts_begin': ts_begin,
            'ts_end': ts_end,
            'limit': limit
        }
        other_topics = [ [t] if isinstance(t, str) else list(t)
                            for t in self.topics ]
        default_alg = options.pop('algorithm', 'nearest')
        options['ts_begin'], options['ts_end'] = self.process_ts(options.get('ts_begin')), self.process_ts(options.get('ts_end'))

        aligner_reader = MsgReader(self.bag, 'aligner', self.aligner_topic)
        target = aligner_reader.get_rows(options['ts_begin'], options['ts_end'])
        
        # modify options slightly before fetching other_topics
        options['ts_begin'] -= 10**9
        options['ts_end'] += 10**9

        # never limit the number of messages of other_topics
        options['limit'] = None

        _others = []
        others = []
        for topic in other_topics:
            if len(topic) == 1:
                topic.append(default_alg)
            _others.append(topic)
        for key, topic in zip(self.topic_keys, self.topics):
            other_reader = MsgReader(self.bag, key, topic)
            others.append((other_reader.get_rows(options['ts_begin'], options['ts_end']), default_alg))
        
        return align(target, *others)

    def fetch_lidar_start_end_ins(self, data):
        for sensor in self.topic_keys:
            if sensor.startswith('lidar') and 'det' not in sensor and 'pose' not in sensor:
                stamp = data[sensor][0]
                _, info = trans_pointcloud2_to_array(data[sensor][1])
                start_packet_stamp = info[0][2]
                end_packet_stamp = info[-1][2]
                start_ins = self.geodetic_msg_to_loc_enu(self.fetch_near(self.topics_desp['gps-0'], start_packet_stamp, 100)[1])
                end_ins = self.geodetic_msg_to_loc_enu(self.fetch_near(self.topics_desp['gps-0'], end_packet_stamp, 100)[1])
                data['{}_start_ins'.format(sensor)] = start_ins
                data['{}_end_ins'.format(sensor)] = end_ins

    def fetch_sensor_ins(self, data):
        gps = data['gps-0'][1]
        v = np.sqrt(gps.position_covariance[4]**2 + gps.position_covariance[5]**2)
        for sensor in self.topic_keys:
            if sensor.startswith('camera') or sensor.startswith('odo') or sensor.startswith('radar'):
                sensor_stamp = data[sensor][1].header.stamp.to_nsec()
                sensor_key = '{}_ins'.format(sensor)

                sensor_inspvax_last = self.fetch_near(self.topics_desp['gps-0'],
                                                    sensor_stamp-5000000, 100)
                sensor_inspvax_next = self.fetch_near(self.topics_desp['gps-0'],
                                                    sensor_stamp+5000000, 100)

                res_pose = self.interpolate_gps(sensor_stamp, sensor_inspvax_last, sensor_inspvax_next)
                inspvax = (sensor_stamp, res_pose)
                data[sensor_key] = inspvax

    def interpolate_gps(self, ts, last_gps, next_gps):
        if last_gps==None and next_gps!=None:
            return self.geodetic_msg_to_loc_enu(next_gps[1])
        elif next_gps==None and last_gps!=None:
            return self.geodetic_msg_to_loc_enu(last_gps[1])
        elif last_gps!=None and next_gps!=None:
            ts /= 1e9
            last_ts = last_gps[1].header.stamp.to_sec()
            next_ts = next_gps[1].header.stamp.to_sec()
            last_pose = self.geodetic_msg_to_loc_enu(last_gps[1])
            next_pose = self.geodetic_msg_to_loc_enu(next_gps[1])
            inter_pose = self.interpolate_pose(ts, last_pose, last_ts, next_pose, next_ts)
            return inter_pose
        else:
            print("[ERROR]NO pose data to interpolate!")
            return None

    @staticmethod
    def interpolate_pose(ref_time, a_p, a_t, b_p, b_t):
        pose_diff = SE3.between(a_p, b_p)
        log_pose = SE3.vector(pose_diff)
        time_offset = (ref_time - a_t) / (b_t - a_t)
        interpolated_pose = log_pose * time_offset
        expmap = SE3.get_pose(interpolated_pose)
        final_pose = SE3.compose(a_p, expmap)
        return final_pose

    def read(self, ts_begin=None, ts_end=None, limit=None):
        data = {'indix': 0}
        for item_key, topic_data in enumerate(self.fectch_align(ts_begin, ts_end, limit)):
            data['indix'] = item_key
            data.update(
                {key: topic_data[i] for i, key in enumerate(self.topic_keys_total)})
            self.fetch_lidar_start_end_ins(data)
            # self.fetch_sensor_ins(data)
            yield data


        
if __name__ == "__main__":
    bag_file = '/home/mouse/Documents/cam2imu_manual_calib/src/cam2imu_manual_calib/cam2imu_manual_calib/total.bag'
    ts_begin = '00:00:00'
    ts_end = '00:00:10'
    base = [31.206388889, 121.689444445]
    aligner = '/camera1/image_raw/compressed'
    topics = {  
        'lidar': '/points_raw',  
        'gps-0': '/met/gps/lla_qj',
        'image': '/camera1/image_raw/compressed'
    }

    bag_reader = RosbagReader(bag_file, base)
    bag_reader.init_topics(aligner, topics)
    for data in bag_reader.read(ts_begin, ts_end):
        print(data['indix'])
        print("lidar_info:------------------------------")
        print((data))
        # 2022-11-14 04:45:22
        # lla_qj = np.array([data['gps-0'][1].latitude, data['gps-0'][1].longitude, data['gps-0'][1].altitude])
        # print(lla_qj)
        # # break
        # print("Image_info-------------")
        # print((data['image'][0]))


        
    # res = bag_reader.fetch_near('/met/gps/lla', 1661980994099910975, 100)
    # print(res[0])
    # print(res[1])
