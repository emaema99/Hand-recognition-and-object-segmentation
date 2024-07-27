import cv2
import numpy as np
import depthai as dai
import time
from scipy.spatial import ConvexHull
import pyny3d.geoms as pyny

import copy

from seg8_hand_utils_v2 import HostSpatialsCalc

mouseX = 0
mouseY = 0
mouse_clicked = False


def get_mouse_pixel_coordinates(event,x,y,flags,param):
    global mouseX,mouseY,mouse_clicked
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print("ciao")
        mouseX,mouseY = x,y
        mouse_clicked = True


if __name__ == '__main__':

    # Create pipeline
    device = dai.Device()
    pipeline = dai.Pipeline()

    # Define a source
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(512, 288)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.setInterleaved(False)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    calib_data = device.readCalibration()
    calib_lens_pos = calib_data.getLensPosition(dai.CameraBoardSocket.CAM_A)
    cam_rgb.initialControl.setManualFocus(calib_lens_pos)

    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left.setCamera("left")
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right.setCamera("right")
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setConfidenceThreshold(255)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setSubpixel(True)

    spatial_location_calculator = pipeline.create(dai.node.SpatialLocationCalculator)
    spatial_location_calculator.inputConfig.setWaitForMessage(True)
    spatial_location_calculator.inputDepth.setBlocking(False)
    spatial_location_calculator.inputDepth.setQueueSize(1)

    # Config
    topLeft = dai.Point2f(0.4, 0.4)
    bottomRight = dai.Point2f(0.6, 0.6)

    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 100
    config.depthThresholds.upperThreshold = 10000
    calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
    config.roi = dai.Rect(topLeft, bottomRight)

    spatial_location_calculator.inputConfig.setWaitForMessage(False)
    spatial_location_calculator.initialConfig.addROI(config)

    # Create output links
    cam_out = pipeline.create(dai.node.XLinkOut)
    cam_out.setStreamName("rgb")
    cam_out.input.setQueueSize(1)
    cam_out.input.setBlocking(False)
    cam_rgb.preview.link(cam_out.input)

    stereo_out = pipeline.create(dai.node.XLinkOut)
    stereo_out.setStreamName("stereo")
    stereo_out.input.setQueueSize(1)
    stereo_out.input.setBlocking(False)
    stereo.depth.link(stereo_out.input)

    xoutSpatialData = pipeline.create(dai.node.XLinkOut)
    xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)
    xoutSpatialData.setStreamName("spatialData")
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
    spatial_location_calculator.passthroughDepth.link(stereo_out.input)
    stereo.depth.link(spatial_location_calculator.inputDepth)

    spatial_location_calculator.out.link(xoutSpatialData.input)
    xinSpatialCalcConfig.out.link(spatial_location_calculator.inputConfig)

    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(spatial_location_calculator.inputDepth)

    # Initialize the area array
    calculated_areas = np.array((0))

    device.startPipeline(pipeline)

    with device:

        # Connect and start the pipeline
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
        stereo_queue = device.getOutputQueue(name="stereo", maxSize=1, blocking=True)
        spatial_calc = HostSpatialsCalc(device, stereo_queue.get())
        spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=1, blocking=True)
        spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

        # mouse callback
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',get_mouse_pixel_coordinates)
        cv2.resizeWindow('image', 1280, 720)

        cv2.namedWindow('depth_image')
        cv2.resizeWindow('depth_image', 1280, 720)

        frame = None
        depth_frame = None
        spatialData = None


        while True:

            for i in range(180):
                rgb_msg = rgb_queue.get()
                frame = copy.deepcopy(rgb_msg.getCvFrame()) if rgb_msg is not None else None
                depth_msg = stereo_queue.get()
                depth_frame = copy.deepcopy(depth_msg.getFrame()) if depth_msg is not None else None
                spatialData = spatialCalcQueue.get().getSpatialLocations()

                cv2.imshow('image', frame)
                cv2.imshow('depth_image', depth_frame)
                cv2.waitKey(10)
                
            if frame is None or depth_frame is None:
                continue

            spatialData = spatialCalcQueue.get().getSpatialLocations()

            cv2.imshow('image', frame)
            cv2.imshow('depth_image', depth_frame)
            cv2.waitKey(5000)

            mouse_coordinates = np.zeros(4, dtype=np.int32)
            while not mouse_clicked:
                time.sleep(0.2)
            mouse_clicked = False
            mouse_coordinates[0] = mouseX
            mouse_coordinates[1] = mouseY
            cv2.circle(frame,(mouseX,mouseY),20,(255,0,0),-1)
            print("mouse_coordinates: ", mouse_coordinates)
            cv2.imshow("image", frame)
            cv2.imshow('depth_image', depth_frame)
            cv2.waitKey(5000)
            while not mouse_clicked:
                time.sleep(0.2)
            mouse_clicked = False
            mouse_coordinates[2] = mouseX
            mouse_coordinates[3] = mouseY
            cv2.circle(frame,(mouseX,mouseY),20,(255,0,0),-1)
            print("mouse_coordinates: ", mouse_coordinates)
            cv2.imshow("image", frame)
            cv2.imshow('depth_image', depth_frame)
            cv2.waitKey(2000)

            margin = 5
            x = min(max(mouse_coordinates[0], margin), depth_frame.shape[1] - margin)
            y = min(max(mouse_coordinates[1], margin), depth_frame.shape[0] - margin)
            roi = (x-margin, y-margin, x+margin, y+margin)
            xmin, ymin, xmax, ymax = roi
            frame[ymin:ymax, xmin:xmax] = frame[ymin:ymax, xmin:xmax] * [0,0,1]
            cv2.imshow("image", frame)
            cv2.waitKey(2000)


            
            point_1_spatials, point_1_pixels = spatial_calc.calc_point_spatial(depth_frame, mouse_coordinates[0:2])
            point_2_spatials, point_2_pixels = spatial_calc.calc_point_spatial(depth_frame, mouse_coordinates[2:4])

            if point_1_spatials is None or point_2_spatials is None:
                print("depth failed")
                continue

            dist = np.linalg.norm(point_1_spatials-point_2_spatials)
            print("dist: ", dist)

            input("")
            
        