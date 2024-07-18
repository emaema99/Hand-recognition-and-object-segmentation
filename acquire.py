import depthai as dai
import numpy as np
import cv2
import pyny3d.geoms as pyny

from seg8_nano_v5 import Seg8
from seg8_hand_utils_v2 import HostSpatialsCalc

black_pixel_threshold = 60

if __name__ == '__main__':

    # Create pipeline
    device = dai.Device()
    pipeline = dai.Pipeline()

    # Define a source
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(1280, 720)
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
    stereo.setSubpixel(False)

    spatial_location_calculator = pipeline.createSpatialLocationCalculator()
    spatial_location_calculator.inputConfig.setWaitForMessage(True)
    spatial_location_calculator.inputDepth.setBlocking(False)
    spatial_location_calculator.inputDepth.setQueueSize(1)

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

    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(spatial_location_calculator.inputDepth)

    # Load the YOLOv8-seg model
    path_to_yolo = "/home/ema/Desktop/depthai-hand-segmentation/models/best_nano8.pt"
    my_seg8 = Seg8(path_to_yolo, True)
    my_seg8.start_threads()

    # Initialize the area array
    calculated_areas = np.array((0))

    device.startPipeline(pipeline)

    with device:
        # Connect and start the pipeline
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        stereo_queue = device.getOutputQueue(name="stereo", maxSize=1, blocking=False)
        spatial_calc = HostSpatialsCalc(device, stereo_queue.get())

        try:
            cv2.namedWindow('Seg8', cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('Seg8', 1280, 720)

            while True:
                rgb_msg = rgb_queue.get()
                frame = rgb_msg.getCvFrame() if rgb_msg is not None else None
                depth_msg = stereo_queue.get()
                depth_frame = depth_msg.getFrame() if depth_msg is not None else None

                # Model prediction
                my_seg8.set_frame_to_seg(frame) 
                seg_result = my_seg8.get_yolo_seg_result() # blocking
                my_seg8.start_seg_post_processing(frame, seg_result)
                annotated_frame, obj_masks_indices, obj_masks_contour_indices, inference_class_list = my_seg8.get_seg_post_processing()
                print(inference_class_list)

                selected_contour_spatials, selected_contour_pixels = spatial_calc.calc_roi_each_point_spatials(depth_frame, obj_masks_contour_indices, 1)

                dark_contour_pixel_indices = []
                current_area = []
                calculated_areas = []

                if len(inference_class_list) > 1:
                    continue

                if inference_class_list is not None and inference_class_list[0] in [0, 2]:
                    if selected_contour_pixels is not None and len(selected_contour_pixels) > 0:
                        selected_contour_pixels = np.array(selected_contour_pixels)
                        # Extract the RGB values at the specified coordinates, assuming selected_contour_pixels is in (x, y) format
                        pixel_values = frame[selected_contour_pixels[:, 1], selected_contour_pixels[:, 0]]
                        # Find indices where pixel intensity is below the black_pixel_threshold (for any RGB channel)
                        dark_contour_pixel_indices = np.argwhere((pixel_values < black_pixel_threshold).any(axis=-1))
                        selected_contour_spatials = selected_contour_spatials[dark_contour_pixel_indices]

                if selected_contour_spatials is not None and len(selected_contour_spatials) > 0:
                    try:
                        selected_contour_spatials_flat = [item for sublist in selected_contour_spatials for item in sublist]
                        selected_contour_spatials_np = np.array(selected_contour_spatials_flat).reshape(-1, 3)
                        polygon = pyny.Polygon(selected_contour_spatials_np)
                        current_area = (polygon.get_area()) / 100
                        print(f'current_area is: {int(current_area)} cm^2')
                    except Exception as e:
                        print(f"Error while processing selected_contour_spatials: {e}")
                
                if current_area is not None:
                    calculated_areas = np.append(calculated_areas, current_area)

                # Show the frame
                cv2.imshow('Seg8', annotated_frame)

                if cv2.waitKey(1) == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Interrupted by user")

        np.save("New_area", calculated_areas)

        my_seg8.stop_threads()
        del my_seg8
        cv2.destroyAllWindows()
