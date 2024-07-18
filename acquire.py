import depthai as dai
import numpy as np
import cv2
import pyny3d.geoms as pyny

from seg8_nano_v5 import Seg8
from seg8_hand_utils_v2 import HostSpatialsCalc

if __name__ == '__main__':

    pipeline = dai.Pipeline()
    device = dai.Device(pipeline)

    # Define a source
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(1280, 720)
    cam_rgb.setInterleaved(False)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.initialControl.setManualFocus(55) # RGB needs fixed focus to properly align with depth

    left = pipeline.createMonoCamera()
    right = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()

    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left.setCamera("left")
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right.setCamera("right")
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setConfidenceThreshold(255)
    stereo.setLeftRightCheck(True) # LR-check is required for depth alignment
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setSubpixel(False)

    spatial_location_calculator = pipeline.createSpatialLocationCalculator()
    spatial_location_calculator.inputConfig.setWaitForMessage(True)

    # Create output
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    stereo_out = pipeline.createXLinkOut()
    stereo_out.setStreamName("stereo")
    stereo.depth.link(stereo_out.input)

    left.out.link(stereo.left) # Connect Left Stero Camera to the Stero Node
    right.out.link(stereo.right) # Connect Right Stero Camera to the Stero Node
    stereo.depth.link(spatial_location_calculator.inputDepth)

    # Connect and start the pipeline
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
    stereo_queue = device.getOutputQueue(name = "stereo", maxSize=1, blocking=True)

    spatial_calc = HostSpatialsCalc(device, stereo_queue.get())

    # Load the YOLOv8-seg model
    path_to_yolo = "/home/ema/Desktop/depthai-hand-segmentation/models/best_nano8.pt"
    my_seg8 = Seg8(path_to_yolo, True)
    my_seg8.start_threads()

    # Initialize the area array
    calculated_areas = np.array((0))

    cv2.namedWindow("Seg8", cv2.WINDOW_NORMAL)

    while True:
        rgb_msg = rgb_queue.get()
        depth_msg = stereo_queue.get()
        if rgb_msg is None and depth_msg is None:
            print("No frame received!")
            continue

        frame = rgb_msg.getCvFrame()
        depth_frame = depth_msg.getFrame()
        if frame is None:
            print("Frame is empty!")
            continue
        elif depth_frame is None:
            print("Depth is empty!")

        print("Frame received. Shape:", frame.shape)

        # Model prediction
        my_seg8.set_frame_to_seg(frame) 
        seg_result = my_seg8.get_yolo_seg_result() # blocking
        my_seg8.start_seg_post_processing(frame, seg_result)
        annotated_frame, obj_masks_indices, obj_masks_contour_indices, inference_class_list = my_seg8.get_seg_post_processing()
        #my_seg8.get_class_names()[
        #my_seg8.get_class_weight()[
        print("Model prediction completed")

        mask_spatial_in_range, mask_pixels_in_range = spatial_calc.calc_roi_each_point_spatials(depth_frame, obj_masks_contour_indices, 1)

        current_area = []
        
        calculated_areas = np.append(calculated_areas, current_area)

        # Show the frame
        cv2.imshow("Seg8", annotated_frame)
        cv2.resizeWindow("Seg8", 800, 600)

        if cv2.waitKey(10) == ord('q'):
            break

# np.save("filename", calculated_areas)

my_seg8.stop_threads()
del my_seg8
cv2.destroyAllWindows()
