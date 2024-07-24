import cv2
import numpy as np
import depthai as dai
import time
from scipy.spatial import ConvexHull
import pyny3d.geoms as pyny

import copy

from seg8_hand_utils_v2 import HostSpatialsCalc

internal_fps = 30
crop = False

if __name__ == '__main__':
    # # Device object declaration
    # device = dai.Device()

    # # ==== Setting Camera Parameters ====
    # # Check USB Speed
    # usb_speed = device.getUsbSpeed()
    # print(f"OAK Connection USB speed: {str(usb_speed).split('.')[-1]}")

    # # Defining OAK internal color camera as source video
    # input_type = "rgb"

    # # Setting Defaul Resolution
    # resolution = (1920, 1080)
    # print("Sensor resolution:", resolution)

    # # Check if the device supports stereo
    # cameras = device.getConnectedCameras()
    # if not(dai.CameraBoardSocket.CAM_B in cameras and dai.CameraBoardSocket.CAM_C in cameras):
    #     print("FATAL ERROR: depth unavailable on this device, ADIOS")
    #     exit(1)

    # # Set Camera FPS for lm_model = lite configuration
    # internal_fps =  internal_fps
    # print(f"Internal camera FPS set to: {internal_fps}")

    # # Used when saving the output in a video file. Should be close to the real fps
    # video_fps = internal_fps

    # # Define Video and Image Size
    # # The internal_frame_height must be set according to input image size of YOLO
    # width, scale_nd = (1920, [1,1])
    # img_h = int(round(resolution[1] * scale_nd[0] / scale_nd[1]))
    # img_w = int(round(resolution[0] * scale_nd[0] / scale_nd[1]))
    # pad_h = (img_w - img_h) // 2
    # pad_w = 0
    # frame_size = img_w
    # crop_w = 0


    # print(f"Internal camera image size: {img_w} x {img_h} - pad_h: {pad_h}")

    # # ==== Define and start pipeline ====

    # # Start defining a pipeline
    # pipeline = dai.Pipeline()
    # pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)
    # pd_input_length = 128

    # # ==== Camera RGB ====
    # print("Creating Color Camera...")
    # cam = pipeline.createColorCamera()
    # cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    # cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    # cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # cam.setInterleaved(False)
    # cam.setIspScale(scale_nd[0], scale_nd[1])
    # cam.setFps(internal_fps)
    # cam.setVideoSize(img_w, img_h)
    # cam.setPreviewSize(img_w, img_h)

    # # ==== Stereo and Depth Camera ====
    # print("Creating MonoCameras, Stereo and SpatialLocationCalculator nodes...")
    # calib_data = device.readCalibration()
    # calib_lens_pos = calib_data.getLensPosition(dai.CameraBoardSocket.CAM_A)
    # print(f"RGB calibration lens position: {calib_lens_pos}")
    # cam.initialControl.setManualFocus(calib_lens_pos) # RGB needs fixed focus to properly align with depth

    # left = pipeline.createMonoCamera()
    # right = pipeline.createMonoCamera()
    # stereo = pipeline.createStereoDepth()

    # left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    # left.setCamera("left")
    # left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    # left.setFps(internal_fps)

    # right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    # right.setCamera("right")
    # right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    # right.setFps(internal_fps)

    # stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # stereo.setConfidenceThreshold(255)
    # stereo.setLeftRightCheck(True) # LR-check is required for depth alignment
    # stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    # stereo.setSubpixel(False) # SubPixel=True brings latency

    # # === Spatial Location Calculator ====
    # spatial_location_calculator = pipeline.createSpatialLocationCalculator()
    # spatial_location_calculator.setWaitForConfigInput(True)
    # spatial_location_calculator.inputDepth.setBlocking(False)
    # spatial_location_calculator.inputDepth.setQueueSize(1)

    # # ==== Pipeline Linking ====
    # print("Pipeline Linking...")

    # # Linking Camera RGB
    # cam_out = pipeline.createXLinkOut()
    # cam_out.setStreamName("RGB_preview")
    # cam_out.input.setQueueSize(1)
    # cam_out.input.setBlocking(False)
    # cam.preview.link(cam_out.input)

    # # Linking Stereo and Depth Camera
    # stereo_out = pipeline.createXLinkOut()
    # stereo_out.setStreamName("stereo")
    # stereo_out.input.setQueueSize(1)
    # stereo_out.input.setBlocking(False)
    # stereo.depth.link(stereo_out.input)

    # left.out.link(stereo.left) # Connect Left Stero Camera to the Stero Node
    # right.out.link(stereo.right) # Connect Right Stero Camera to the Stero Node
    # stereo.depth.link(spatial_location_calculator.inputDepth)


    # device.startPipeline(pipeline)

    # print("Pipeline created.")

    # # Define data queues
    # rgb_queue = device.getOutputQueue(name = "RGB_preview", maxSize=1, blocking=False)
    # stereo_queue = device.getOutputQueue(name = "stereo", maxSize=1, blocking=False)




    # # Connect and start the pipeline
    # spatial_calc = HostSpatialsCalc(device, stereo_queue.get())





    # Create pipeline
    device = dai.Device()
    pipeline = dai.Pipeline()

    cam = pipeline.createColorCamera()
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setPreviewSize(512, 288)
    cam.setInterleaved(False)
    #cam.setIspScale(scale_nd[0], scale_nd[1])
    cam.setFps(internal_fps)
    #cam.setVideoSize(img_w, img_h)
    #cam.setPreviewSize(img_w, img_h)
    
    

    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    # Properties
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    #monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    #monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True) # LR-check is required for depth alignment
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.initialConfig.setConfidenceThreshold(255)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)
    stereo.setOutputSize(512,288)

    # Linking
    # Create output links
    cam_out = pipeline.create(dai.node.XLinkOut)
    cam_out.setStreamName("rgb")
    cam_out.input.setQueueSize(1)
    cam_out.input.setBlocking(False)
    cam.preview.link(cam_out.input)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutDepth.setStreamName("depth")
    stereo.depth.link(xoutDepth.input)

    # mouse callback
    cv2.namedWindow('image')
    cv2.resizeWindow('image', 1280, 720)

    cv2.namedWindow('depth_image')
    cv2.resizeWindow('depth_image', 1280, 720)

    frame = None
    depth_frame = None
    spatialData = None

    device.startPipeline(pipeline)

    # Connect to device and start pipeline
    #with dai.Device(pipeline) as device:

    # Output queue will be used to get the depth frames from the outputs defined above
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
    depth_queue = device.getOutputQueue(name="depth", maxSize=1, blocking=True)


    while True:


        rgb_msg = rgb_queue.get()
        frame = copy.deepcopy(rgb_msg.getCvFrame()) if rgb_msg is not None else None
        depth_msg = depth_queue.get()
        depth_frame = copy.deepcopy(depth_msg.getFrame()) if depth_msg is not None else None

        if frame is None or depth_frame is None:
            continue

        pixel_y = int(frame.shape[0]/2)
        pixel_x = int(frame.shape[1]/2)
        print("pixel: ", pixel_x, pixel_y)
        print("pixel depth: ", np.median(depth_frame[pixel_x-50:pixel_x+50, pixel_y-50:pixel_y+50].reshape(-1), axis=0))

        frame[pixel_x-50:pixel_x+50, pixel_y-50:pixel_y+50] = frame[pixel_x-50:pixel_x+50, pixel_y-50:pixel_y+50] * [0,0,1]

        cv2.imshow('image', frame)
        cv2.imshow('depth_image', depth_frame)
        cv2.waitKey(10)
