#!/usr/bin/env python3

import cv2
import numpy as np
import time
import copy

from HandTrackerRendererV3 import HandTrackerRenderer
from CustomPipeline2 import CustomPipeline
from HandTrackerEdgeV4 import HandTracker
from seg8_nano_v4 import Seg8
from seg8_hand_utils import HostSpatialsCalc
from arduino_wifi_communication import ArduinoCommunicator

'''
Code for running YOLOV8 segmentation models on Jetson Nano 4GB and gesture recognition (hand tracking) on OAK devices.
Based on https://github.com/geaxgx/depthai_hand_tracker and on the works of Ultralytics https://docs.ultralytics.com/tasks/segment/.
'''

path_to_yolo = "/home/ema/Desktop/depthai-hand-segmentation/models/best_nano8.pt"
num_elements_moving_avarage = 15
hand_obj_dist_threshold = 250 #[mm]
down_sampling = 5

EXO_COMM = False

def add_grid_img(gesture_status, object_name, object_weight, fps, grid_height, grid_width, cell_height, cell_width):
    '''
    It creates a blank image grid with grasping status, object name, object weight, and FPS displayed.
    '''
    grid_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Adds fixed text labels
    cv2.putText(grid_img, f'Status:', (10, cell_height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_img, f'Object:', (2*cell_width-30, cell_height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_img, f'Weight:', (10, cell_height + cell_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_img, f'Fps:', (2*cell_width - 30, cell_height + cell_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # Adds dynamic data. Modify the blank spaces and font dimension to obtain a better visualization. 
    cv2.putText(grid_img, ' ' + gesture_status, (90, cell_height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_img, object_name, (2*cell_width + 60, cell_height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_img, ' ' + object_weight +' kg', (90, cell_height + cell_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_img, f'{fps}', (2*cell_width + 60, cell_height + cell_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    return grid_img

def add_grid_img2(grasping_status, grasped_obj_name, obj_weight, fps, grid_height, grid_width, cell_height, cell_width):
    '''
    It determines the grasping status and converts the object weight and FPS to strings and it prints them in the grid below the video preview.
    NOTE: grasping status is when there are 3 or more fingers closed, near the object (based on the palm distance from the object's mask pixels).
    '''
    if grasping_status:
        gesture_text = "Grasping"
    else:
        gesture_text = "Not Grasping"

    if obj_weight is None:
        obj_weight = "-"

    # Casting float values into integers to delete decimal values.
    fps = int(float(fps))

    return add_grid_img(gesture_text, grasped_obj_name, str(obj_weight), str(fps), grid_height, grid_width, cell_height, cell_width)

def calc_fps(start_time, frame_count, fps):
    '''
    Calculates the fps of the preview, checking if more than one second has passed since the last update.
    '''
    frame_count += 1

    if time.time() - start_time > 1:
        fps = frame_count / (time.time() - start_time)
        start_time = time.time()
        frame_count = 0

    return start_time, frame_count, fps

def draw_mask(frame_to_annotate, mask, class_type):
    '''
    Mask appliance based on the class of the detected object. 
    Using a copy of the frame not to alter the original.
    '''
    img = np.copy(frame_to_annotate)
    annotated_frame = np.copy(frame_to_annotate)

    if class_type == 0:
        mask_color = [0.1, 0.1, 1]
    elif class_type == 1:
        mask_color = [1, 0.1, 0.1]
    else:
        mask_color = [0.1, 1, 0.1]

    annotated_frame[mask[:,0], mask[:,1], :] = img[mask[:,0], mask[:,1], :] * mask_color

    return annotated_frame

if __name__ == '__main__':
    # Initialize the pipeline (while setting the thresholds for the palm detection and landmark detection algorithms),
    # the hand tracker (with frame size and padding parameters from the pipeline) and the hand tracker renderer.
    pipeline = CustomPipeline(pd_score_thresh = 0.4, lm_score_thresh = 0.4)
    handTracker = HandTracker(pipeline.frame_size, pipeline.pad_h, pipeline.pad_w, pipeline.lm_score_thresh)
    handRender = HandTrackerRenderer(handTracker)
    spatial_calc = HostSpatialsCalc(pipeline.getDevice())

    # Initialize and start the YOLO segmentation model
    my_seg8 = Seg8(path_to_yolo)
    my_seg8.start_threads()

    # Initialize variables for frame processing and FPS calculation
    first_frame = True
    start_time = time.time()
    frame_count = 0
    fps = 0

    # Initialize arrays and variables for grasping detection and object tracking
    grasping_status_arr = np.zeros(num_elements_moving_avarage, dtype=np.bool_)
    hand_spatial_arr = np.zeros([num_elements_moving_avarage, 3])
    obj_spatial_arr = np.zeros([num_elements_moving_avarage, 3])
    last_grasped_obj_name = "-"
    last_obj_weight = "-"
    released = True
    moving_avarage_index = 0

    depth_data = pipeline.getDepthData() # used to retrieve HFOV info

    # Initialize communication with the exosuit (if enabled).
    if EXO_COMM:
        communicator = ArduinoCommunicator(ip='192.168.0.21', port=50000)
        communicator.connect()

    try:
        while True:
            # Capture depth and RGB frame from the camera pipeline.
            frame = pipeline.getFrame()
            depth_frame = pipeline.getDepthFrame() # UINT16 - mm

            if first_frame:
                # Sets the grid dimensions, depending on the frame shape.
                grid_height = 100
                grid_width = frame.shape[1]
                grid_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
                # Define the cell size for the grid.
                cell_height = grid_height // 2
                cell_width = grid_width // 3

                # This "first frame" condition ensures the correct visualization (otherwise we can't see anything: conda environment problem related to my PC).
                grid_img = add_grid_img("Not Grasping", "-", "-", "0", grid_height, grid_width, cell_height, cell_width)
                combined_img = np.vstack((frame, grid_img))

                # Here the frame is resized for better visualization, while mantaining the ratio.
                cv2.namedWindow('Object segmentation & grasping detection', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('Object segmentation & grasping detection', combined_img)
                cv2.resizeWindow('Object segmentation & grasping detection', 1280, 720)
                cv2.waitKey(1)
                # Exit from the "first frame" phase.
                first_frame = False

            # Update the segmentation model with current frame and depth frame
            my_seg8.update(frame, depth_frame)

            # Initialization of variables for frame annotation.
            frame_final = np.copy(frame)
            obj_spatial = None
            hand = None
            class_names = None
            obj_weight = None
            grasping_status = False
            grasped_obj_name = "-"
            masks_spatial = []
            hand_obj_dist = "-"

            # Perform hand tracker inference.
            handsData = handTracker.getHandsData(pipeline.getManagerScriptOutput())

            if handsData:
                hand = copy.deepcopy(handsData[0])
                grasping_status = hand.is_grasping
                hand_spatial = hand.xyz

            # Perform object segmentation inference
            annotated_frame = copy.deepcopy(my_seg8.get_annotated_frame())
            masks_indices = copy.deepcopy(my_seg8.get_obj_masks_indices())
            result_class_list = copy.deepcopy(my_seg8.get_result_class_list())

            if annotated_frame is not None:
                # Calculate FPS.
                start_time, frame_count, fps = calc_fps(start_time, frame_count, fps)

                for i in range(len(masks_indices)):
                    # Calculate spatial data for each mask
                    mask_spatial = spatial_calc.calc_roi_each_point_spatials(depth_frame, depth_data, masks_indices[i], down_sampling)

                    if isinstance(mask_spatial, np.ndarray):
                        mask_spatial = mask_spatial.reshape(-1,3)

                    masks_spatial.append(mask_spatial)

                # Process hand data (if available and if a hand was detected)
                if handsData and hand_spatial is not None:
                    if grasping_status and hand_spatial is not None:
                        if len(masks_spatial) > 0:
                            min_dist = np.inf
                            min_dist_index = None
                            min_index = None
                            for i in range(len(masks_spatial)):
                                if masks_spatial[i] is not None and isinstance(masks_spatial[i], np.ndarray):
                                    hand_spatial_matr = np.full(masks_spatial[i].shape, hand_spatial)
                                    # Calculates the distance between the palm and the closest pixel of the mask.
                                    dists = np.linalg.norm(hand_spatial_matr - masks_spatial[i], axis = 1)
                                    mask_min_dist = np.min(dists)
                                    if mask_min_dist < min_dist:
                                        min_dist = copy.deepcopy(mask_min_dist)
                                        min_index = copy.deepcopy(i)
                                        mask_min_dist_index = np.argmin(dists)
                                        min_dist_index = copy.deepcopy(mask_min_dist_index)

                            if min_index is not None and min_index >= 0 and min_index < len(result_class_list):
                                grasped_obj_name = my_seg8.get_class_names()[result_class_list[min_index]]
                                obj_weight = my_seg8.get_class_weight()[result_class_list[min_index]]
                                hand_obj_dist = int(min_dist)

                                # Update moving average arrays with current data.
                                hand_spatial_arr[moving_avarage_index, :] = hand_spatial
                                obj_spatial_arr[moving_avarage_index, :] = masks_spatial[min_index][min_dist_index,:]

                                # Calculate averages for grasping decision.
                                hand_spatial_arr_without_nan_index = np.argwhere(~np.isnan(hand_spatial_arr)).reshape(-1)

                                if hand_spatial_arr_without_nan_index.shape[0] > 0:
                                    hand_spatial_arr_without_nan = hand_spatial_arr[hand_spatial_arr_without_nan_index]
                                    hand_spatial_avarage = np.mean(hand_spatial_arr_without_nan, axis=0)

                                obj_spatial_arr_without_nan_index = np.argwhere(~np.isnan(obj_spatial_arr)).reshape(-1)

                                # Deals with NaN values.
                                if obj_spatial_arr_without_nan_index.shape[0] > 0:
                                    obj_spatial_arr_without_nan = obj_spatial_arr[obj_spatial_arr_without_nan_index]
                                    obj_spatial_avarage = np.mean(obj_spatial_arr_without_nan, axis=0)

                                hand_obj_dist = np.linalg.norm(hand_spatial_avarage-obj_spatial_avarage)
                                print("hand_obj_dist: ", hand_obj_dist)

                                if hand_obj_dist < hand_obj_dist_threshold:
                                    grasping_status_arr[moving_avarage_index] = True
                                else:
                                    grasping_status_arr[moving_avarage_index] = False

                                # Update moving average index.
                                moving_avarage_index = (moving_avarage_index + 1) % num_elements_moving_avarage

                    else:
                        # Handle case when hand is not grasping.
                        grasping_status_arr[moving_avarage_index] = False
                        hand_spatial_arr[moving_avarage_index, :] = [np.nan, np.nan, np.nan]
                        obj_spatial_arr[moving_avarage_index, :] = [np.nan, np.nan, np.nan]
                        
                        # Update moving average index.
                        moving_avarage_index = (moving_avarage_index + 1) % num_elements_moving_avarage

                # Draw hand annotations on the frame.
                frame_final = handRender.draw(annotated_frame, handsData)

                # Calculate the moving average grasping status.
                if np.sum(grasping_status_arr) > num_elements_moving_avarage/2:
                    if released:
                        last_grasped_obj_name = grasped_obj_name
                        last_obj_weight = obj_weight
                        # If the communication is ON, sends the weight to the exosuit.
                        if EXO_COMM:
                            communicator.send_weight(copy.deepcopy(obj_weight))
                    released = False
                    grasping_status_avarage = True

                else:
                    if EXO_COMM:
                        communicator.send_weight(0)
                    released = True
                    last_grasped_obj_name = "-"
                    last_obj_weight = "-"
                    grasping_status_avarage = False 

                # Update the grid image with current status.
                grid_img = add_grid_img2(grasping_status_avarage, last_grasped_obj_name, last_obj_weight, fps, grid_height, grid_width, cell_height, cell_width)

                # Stack the main frame and the grid image vertically.
                combined_img = np.vstack((frame_final, grid_img))

                cv2.namedWindow('Object segmentation & grasping detection', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('Object segmentation & grasping detection', combined_img)
                cv2.resizeWindow('Object segmentation & grasping detection', 1280, 720)

            # Break the loop if 'q' key is pressed.
            if cv2.waitKey(10) == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    # Send weight = 0 to the communicator if EXO_COMM is enabled.
    if EXO_COMM:
        communicator.send_weight(0)

    # Stop the segmentation threads, close pipeline and release resources.
    my_seg8.stop_threads()
    pipeline.exit()
    del pipeline
    del my_seg8
    cv2.destroyAllWindows()
