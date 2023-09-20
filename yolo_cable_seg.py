#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class CableDetectionNode:
    def __init__(self):
        # ROS node
        rospy.init_node('cable_detection_node')
        # Subscribe
        self.image_sub = rospy.Subscriber('/input/image', Image, self.image_callback)
        # Publish
        self.masked_image_pub = rospy.Publisher('/output/masked_image', Image, queue_size=10)
        
        # Load model
        self.model = YOLO('weight/yolo_cable.pt')
        
        # Create a CV Bridge
        self.bridge = CvBridge()
    
    # Predict mask from ROS Image message
    def image_callback(self, data):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        
        # Run inference
        results = self.model(cv_image)
        
        # plot result with mask only option
        for r in results:
            im_array = r.plot(
                labels=False,
                boxes=False,
                masks=True
            )
            # Convert image array back to OpenCV format
            cv_masked_image = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
            
            # Publish the masked image
            self.masked_image_pub.publish(self.bridge.cv2_to_imgmsg(cv_masked_image, "bgr8"))
        
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = CableDetectionNode()
    node.run()
