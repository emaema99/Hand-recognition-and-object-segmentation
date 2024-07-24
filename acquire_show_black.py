import depthai as dai
import numpy as np
import cv2
import time

from scipy.spatial import ConvexHull
import pyny3d.geoms as pyny
from seg8_nano_v5 import Seg8
from seg8_hand_utils_v3 import HostSpatialsCalc

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

black_pixel_threshold = 45

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
    stereo.setSubpixel(False)
    stereo.setOutputSize(512,288)

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
            cv2.namedWindow('Dark Pixels', cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('Dark Pixels', 1280, 720)
            cv2.namedWindow('extremetes_pixel_frame', cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('extremetes_pixel_frame', 1280, 720)

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

                try:
                    selected_spatials, selected_pixels = spatial_calc.calc_roi_each_point_spatials(depth_frame, obj_masks_indices, 1)
                    if selected_spatials is None or selected_pixels is None:
                        print("Warning: calc_roi_each_point_spatials returned None values")
                        selected_spatials, selected_pixels = [], []
                except Exception as e:
                    print(f"Error in calc_roi_each_point_spatials")
                    selected_spatials, selected_pixels = [], []

                dark_pixel_indices = None
                current_area = None
                area_threshold = 700 # cm^2
                
                dark_pixel_frame = frame.copy()
                extremetes_pixel_frame = frame.copy()

                # if inference_class_list and len(inference_class_list) > 0 and inference_class_list[0] in [0, 2]:
                #     if selected_pixels is not None and len(selected_pixels) > 0:
                #         area_threshold = 150 # cm^2
                #         selected_pixels = np.array(selected_pixels)
                #         pixel_values = frame[selected_pixels[:, 0], selected_pixels[:, 1]]
                #         dark_pixel_indices = np.argwhere((pixel_values < black_pixel_threshold).any(axis=-1))
                #         selected_spatials = selected_spatials[dark_pixel_indices]

                #         # Highlight dark pixels in the dark_pixel_frame
                #         for i in dark_pixel_indices:
                #             cv2.circle(dark_pixel_frame, (selected_pixels[i[0], 1], selected_pixels[i[0], 0]), 1, (0, 255, 0), -1)

                if selected_spatials is not None and len(selected_spatials) > 0:
                    try:
                        selected_spatials = np.reshape(selected_spatials, (selected_spatials.shape[0], 3))
                        min_x_index = np.argmin(selected_spatials[:,0])
                        point1 = selected_spatials[min_x_index, :]
                        max_x_index = np.argmax(selected_spatials[:,0])
                        point2 = selected_spatials[max_x_index, :]
                        min_y_index = np.argmin(selected_spatials[:,1])
                        point3 = selected_spatials[min_y_index, :]
                        max_y_index = np.argmax(selected_spatials[:,1])
                        point4 = selected_spatials[max_y_index, :]

                        print("X max: ",point2)
                        print("X min: ",point1)

                        print("\nH length: ",point4[1]-point3[1])
                        print("W length: ",point2[0]-point1[0])

                        # Calculate the object's distance using the depth frame
                        obj_z = np.quantile(selected_spatials[:,2], 0.8)
                        for i in range(selected_spatials.shape[0]):
                            selected_spatials[i,2] = obj_z

                        selected_spatials_flat = [item for sublist in selected_spatials for item in sublist]
                        selected_spatials_np = np.array(selected_spatials_flat).reshape(-1, 3)
                        # polygon = ConvexHull(selected_spatials_np)
                        polygon = pyny.Polygon(selected_spatials)
                        # current_area = (polygon.area) / 100
                        current_area = (polygon.get_area()) / 100
                        
                        print("plot type: ", type(polygon.get_plotable3d()))
                        print("plot: ", polygon.points)

                        indexes = []
                        for i in range(selected_spatials.shape[0]):
                            for point in polygon.points:
                                if selected_spatials[i, 0]==point[0] and selected_spatials[i, 1]==point[1] and selected_spatials[i, 2]==point[2]:
                                    indexes.append(i)
                        
                        
                        selected_pixels_poly_contour_y = selected_pixels[indexes,0]
                        selected_pixels_poly_contour_x = selected_pixels[indexes,1]
                        selected_pixels_poly_contour_xy = selected_pixels[indexes,:]
                        selected_pixels_poly_contour_xy[:,0] = selected_pixels_poly_contour_x
                        selected_pixels_poly_contour_xy[:,1] = selected_pixels_poly_contour_y
                        cv2.drawContours(extremetes_pixel_frame, [selected_pixels_poly_contour_xy], -1, (255), cv2.FILLED)
                        # for index in indexes:
                        #     cv2.circle(extremetes_pixel_frame, (selected_pixels[index, 1], selected_pixels[index, 0]), 1, (0, 255, 0), -1)
                        
                        # cv2.draw(polygon.plot2d())
                        print(f'\ncurrent_area is: {int(current_area)} cm^2')
                        # cv2.circle(extremetes_pixel_frame, (selected_pixels[min_x_index, 1], selected_pixels[min_x_index, 0]), 1, (0, 255, 0), -1)
                        # cv2.circle(extremetes_pixel_frame, (selected_pixels[max_x_index, 1], selected_pixels[max_x_index, 0]), 1, (0, 255, 0), -1)
                        # cv2.circle(extremetes_pixel_frame, (selected_pixels[min_y_index, 1], selected_pixels[min_y_index, 0]), 1, (0, 255, 0), -1)
                        # cv2.circle(extremetes_pixel_frame, (selected_pixels[max_y_index, 1], selected_pixels[max_y_index, 0]), 1, (0, 255, 0), -1)
                        # for i in range(selected_pixels.shape[0]): 
                        #     cv2.circle(extremetes_pixel_frame, (selected_pixels[i, 1], selected_pixels[i, 0]), 1, (0, 255, 0), -1)
                    except Exception as e:
                        print(f"Error while processing selected_spatials: {e}")

                # # metodo stupido
                # if selected_spatials is not None and len(selected_spatials) > 0:
                #     try:
                #         selected_spatials = np.reshape(selected_spatials, (selected_spatials.shape[0], 3))
                #         min_x_index = np.argmin(selected_spatials[:,0])
                #         point1 = selected_spatials[min_x_index, :]
                #         max_x_index = np.argmax(selected_spatials[:,0])
                #         point2 = selected_spatials[max_x_index, :]
                #         min_y_index = np.argmin(selected_spatials[:,1])
                #         point3 = selected_spatials[min_y_index, :]
                #         max_y_index = np.argmax(selected_spatials[:,1])
                #         point4 = selected_spatials[max_y_index, :]

                #         cv2.circle(extremetes_pixel_frame, (selected_pixels[min_x_index, 1], selected_pixels[min_x_index, 0]), 1, (0, 255, 0), -1)
                #         cv2.circle(extremetes_pixel_frame, (selected_pixels[max_x_index, 1], selected_pixels[max_x_index, 0]), 1, (0, 255, 0), -1)
                #         cv2.circle(extremetes_pixel_frame, (selected_pixels[min_y_index, 1], selected_pixels[min_y_index, 0]), 1, (0, 255, 0), -1)
                #         cv2.circle(extremetes_pixel_frame, (selected_pixels[max_y_index, 1], selected_pixels[max_y_index, 0]), 1, (0, 255, 0), -1)

                #         points = np.array([point1, point2, point3, point4])
                #         polygon = ConvexHull(points)
                #         # polygon = pyny.Polygon(selected_spatials_np)
                #         current_area = (polygon.area) / 100
                #         # current_area = (polygon.get_area()) / 100
                #         print(f'current_area is: {int(current_area)} cm^2')
                #         selected_spatials = None

                #     except Exception as e:
                #         print(f"Error while processing selected_spatials: ", e)
                #         current_area = None

                if isinstance(current_area, (int,float)) and current_area < area_threshold:
                    calculated_areas = np.append(calculated_areas, current_area)

                cv2.imshow('Seg8', annotated_frame)
                cv2.imshow('Dark Pixels', dark_pixel_frame)
                cv2.imshow('extremetes_pixel_frame', extremetes_pixel_frame)
                time.sleep(0.5)

                if cv2.waitKey(1) == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Interrupted by user")

    np.save("New_area.npy", calculated_areas)

    areas = np.load('New_area.npy')
    print(areas)
    mean_areas = np.mean(areas)
    print("\nMean area: ", mean_areas, "\n")

    my_seg8.stop_threads()
    del my_seg8
    cv2.destroyAllWindows()
