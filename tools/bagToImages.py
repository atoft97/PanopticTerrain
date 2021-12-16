#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


parser = argparse.ArgumentParser(description='Extract images from a ROS bag.')
parser.add_argument("bag_file", help="Input ROS bag.")
parser.add_argument("output_dir", help="Output directory.")
parser.add_argument("image_topic", help="Image topic.")

args = parser.parse_args()

print("Extract images from %s on topic %s into %s" % (args.bag_file, args.image_topic, args.output_dir))

bag = rosbag.Bag(args.bag_file, "r")
bridge = CvBridge()
count = 0


topics = bag.get_type_and_topic_info()[1].keys()
for topic in topics:
    print(topic)

for topic, msg, t in bag.read_messages(topics="/ugv_sensors/camera/color/image/compressed"):
    #print(topic)
    #print(dir(msg))
    #print(type(msg))
    cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

    cv2.imwrite("/home/potetsos/skule/2021Host/prosjekt/dataFFI/ffiBilder/" + "roadDrive2" + "/frame" + str(count).zfill(5) +".png", cv_img)
    print("Wrote image %i" % count)

    count += 1


bag.close()

