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
Code for running YOLOV8-seg models on Jetson Nano 4GB and gesture recognition (hand tracking) on OAK devices.
Based on https://github.com/tirandazi/depthai-yolov8-segment and https://github.com/geaxgx/depthai_hand_tracker.
'''

path_to_yolo = "/home/ema/Desktop/depthai-hand-segmentation/models/best_nano8.pt"
num_elements_moving_avarage = 15
hand_obj_dist_threshold = 250 #[mm]
down_sampling = 5

EXO_COMM = True

def add_grid_img(gesture_status, object_name, object_weight, hand_obj_dist, fps, grid_height, grid_width, cell_height, cell_width):
    grid_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # FIXED
    cv2.putText(grid_img, f'Status:', (10, cell_height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_img, f'Object:', (2*cell_width-30, cell_height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_img, f'Weight:', (10, cell_height + cell_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_img, f'Fps:', (2*cell_width - 30, cell_height + cell_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # DATA
    cv2.putText(grid_img, ' ' + gesture_status, (90, cell_height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_img, object_name, (2*cell_width + 60, cell_height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_img, ' ' + object_weight +' kg', (90, cell_height + cell_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(grid_img, f'{fps}', (2*cell_width + 60, cell_height + cell_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    return grid_img

def add_grid_img2(grasping_status, grasped_obj_name, obj_weight, hand_obj_dist, fps, grid_height, grid_width, cell_height, cell_width):
    # Drawing constant text elements
    if grasping_status:
        gesture_text = "Grasping"
    else:
        gesture_text = "Not Grasping"

    if obj_weight is None:
        obj_weight = "-"

    # # check type of hand_obj_dist and if not a string ("-") cast it into an integer
    # if hand_obj_dist is not None and not type(hand_obj_dist)==str and not np.isnan(hand_obj_dist):
    #     hand_obj_dist = int(hand_obj_dist) / 10 # convert in cm
    # else:
    #     hand_obj_dist = "-"
    
    fps = int(float(fps))

    return add_grid_img(gesture_text, grasped_obj_name, str(obj_weight), str(hand_obj_dist), str(fps), grid_height, grid_width, cell_height, cell_width)

def calc_fps(start_time, frame_count, fps):
    frame_count += 1
    if time.time() - start_time > 1:
        fps = frame_count / (time.time() - start_time)
        start_time = time.time()
        frame_count = 0
    return start_time, frame_count, fps

def draw_mask(frame_to_annotate, mask, class_type):
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
    pipeline = CustomPipeline(pd_score_thresh = 0.4, lm_score_thresh = 0.4)
    handTracker = HandTracker(pipeline.frame_size, pipeline.pad_h, pipeline.pad_w, pipeline.lm_score_thresh)
    handRender = HandTrackerRenderer(handTracker)
    spatial_calc = HostSpatialsCalc(pipeline.getDevice())

    my_seg8 = Seg8(path_to_yolo)
    my_seg8.start_threads()

    first_frame = True
    start_time = time.time()
    frame_count = 0
    fps = 0

    grasping_status_arr = np.zeros(num_elements_moving_avarage, dtype=np.bool_)
    hand_spatial_arr = np.zeros([num_elements_moving_avarage, 3])
    obj_spatial_arr = np.zeros([num_elements_moving_avarage, 3])
    last_grasped_obj_name = "-"
    last_obj_weight = "-"
    released = True
    moving_avarage_index = 0

    if EXO_COMM:
        communicator = ArduinoCommunicator(ip='192.168.0.21', port=50000)
        communicator.connect()

    try:
        while True:
            frame = pipeline.getFrame()
            depth_frame = pipeline.getDepthFrame()

            if first_frame:
                # blank image for the grid
                grid_height = 100
                grid_width = frame.shape[1]
                grid_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
                # define cell size
                cell_height = grid_height // 2
                cell_width = grid_width // 3

                # this condition ensures the correct visualization (otherwise we can't see anything -> conda env related)
                grid_img = add_grid_img("Not Grasping", "-", "-", "-", "0", grid_height, grid_width, cell_height, cell_width)
                combined_img = np.vstack((frame, grid_img))
                cv2.namedWindow('Object segmentation & grasping detection', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('Object segmentation & grasping detection', combined_img)
                cv2.resizeWindow('Object segmentation & grasping detection', 1280, 720)
                cv2.waitKey(1)
                first_frame = False

            my_seg8.update(frame, depth_frame)

            frame_final = np.copy(frame)
            obj_spatial = None
            hand = None
            class_names = None
            obj_weight = None

            grasping_status = False
            grasped_obj_name = "-"
            masks_spatial = []
            hand_obj_dist = "-"
            
            # hand tracker inference
            handsData = handTracker.getHandsData(pipeline.getManagerScriptOutput())
            if handsData:
                hand = copy.deepcopy(handsData[0])
                grasping_status = hand.is_grasping
                hand_spatial = hand.xyz

            # seg8 inference
            annotated_frame = copy.deepcopy(my_seg8.get_annotated_frame())
            masks_indices = copy.deepcopy(my_seg8.get_obj_masks_indices())
            result_class_list = copy.deepcopy(my_seg8.get_result_class_list())

            if annotated_frame is not None:
                # annotated_frame = copy.deepcopy(frame)
                start_time, frame_count, fps = calc_fps(start_time, frame_count, fps)

                for i in range(len(masks_indices)):
                    #mask_spatial, centroid = spatial_calc.calc_roi_spatials(depth_frame, masks_indices[i])
                    #mask_spatial, roi_pixels_in_range_downsampled = spatial_calc.calc_roi_each_point_spatials(depth_frame, masks_indices[i], down_sampling)
                    mask_spatial = spatial_calc.calc_roi_each_point_spatials(depth_frame, masks_indices[i], down_sampling)
                    if isinstance(mask_spatial, np.ndarray):
                        mask_spatial = mask_spatial.reshape(-1,3)
                    masks_spatial.append(mask_spatial)
                    # if roi_pixels_in_range_downsampled is not None:
                    #     print("ciao")
                    #     annotated_frame = draw_mask(annotated_frame, roi_pixels_in_range_downsampled, result_class_list[i])

                if handsData and hand_spatial is not None:
                    if grasping_status and hand_spatial is not None:
                        if len(masks_spatial) > 0:
                            min_dist = np.inf
                            min_dist_index = None
                            min_index = None
                            for i in range(len(masks_spatial)):
                                if masks_spatial[i] is not None and isinstance(masks_spatial[i], np.ndarray):
                                    hand_spatial_matr = np.full(masks_spatial[i].shape, hand_spatial)
                                    dists = np.linalg.norm(hand_spatial_matr-masks_spatial[i], axis=1)
                                    mask_min_dist = np.min(dists)
                                    if mask_min_dist < min_dist:
                                        min_dist = copy.deepcopy(mask_min_dist)
                                        min_index = copy.deepcopy(i)
                                        mask_min_dist_index = np.argmin(dists)
                                        min_dist_index = copy.deepcopy(mask_min_dist_index)

                            if min_index is not None and min_index >= 0 and min_index < len(result_class_list):
                                # TODO: capire perche' a volte min_index va oltre l'indice
                                grasped_obj_name = my_seg8.get_class_names()[result_class_list[min_index]]
                                obj_weight = my_seg8.get_class_weight()[result_class_list[min_index]]
                                hand_obj_dist = int(min_dist)

                                # aggiungo i dati alla media mobile positivamente
                                hand_spatial_arr[moving_avarage_index, :] = hand_spatial
                                obj_spatial_arr[moving_avarage_index, :] = masks_spatial[min_index][min_dist_index,:]

                                # computo le medie per calcolare se è grasping
                                hand_spatial_arr_without_nan_index = np.argwhere(~np.isnan(hand_spatial_arr)).reshape(-1)
                                if hand_spatial_arr_without_nan_index.shape[0] > 0:
                                    hand_spatial_arr_without_nan = hand_spatial_arr[hand_spatial_arr_without_nan_index]
                                    hand_spatial_avarage = np.mean(hand_spatial_arr_without_nan, axis=0)
                                obj_spatial_arr_without_nan_index = np.argwhere(~np.isnan(obj_spatial_arr)).reshape(-1)
                                if obj_spatial_arr_without_nan_index.shape[0] > 0:
                                    obj_spatial_arr_without_nan = obj_spatial_arr[obj_spatial_arr_without_nan_index]
                                    obj_spatial_avarage = np.mean(obj_spatial_arr_without_nan, axis=0)

                                hand_obj_dist = np.linalg.norm(hand_spatial_avarage-obj_spatial_avarage)
                                print("hand_obj_dist: ", hand_obj_dist)

                                if hand_obj_dist < hand_obj_dist_threshold:
                                    grasping_status_arr[moving_avarage_index] = True
                                else:
                                    grasping_status_arr[moving_avarage_index] = False

                                # aggiorno l'indice della media mobile
                                moving_avarage_index = (moving_avarage_index + 1) % num_elements_moving_avarage

                    else:
                        # annotare che la mano non ha nulla in mano
                        grasping_status_arr[moving_avarage_index] = False
                        hand_spatial_arr[moving_avarage_index, :] = [np.nan, np.nan, np.nan]
                        obj_spatial_arr[moving_avarage_index, :] = [np.nan, np.nan, np.nan]
                        # aggiorno l'indice della media mobile
                        moving_avarage_index = (moving_avarage_index + 1) % num_elements_moving_avarage

                frame_final = handRender.draw(annotated_frame, handsData) # simply draws the hand (repo) 

                # faccio la media dei valori della media mobile
                if np.sum(grasping_status_arr) > num_elements_moving_avarage/2:
                    if released:
                        last_grasped_obj_name = grasped_obj_name
                        last_obj_weight = obj_weight
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

                grid_img = add_grid_img2(grasping_status_avarage, last_grasped_obj_name, last_obj_weight, hand_obj_dist, fps, grid_height, grid_width, cell_height, cell_width)

                # Stack the main frame and the grid image vertically
                combined_img = np.vstack((frame_final, grid_img))
                cv2.namedWindow('Object segmentation & grasping detection', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('Object segmentation & grasping detection', combined_img)
                cv2.resizeWindow('Object segmentation & grasping detection', 1280, 720)

            key = cv2.waitKey(10)
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    if EXO_COMM:
        communicator.send_weight(0)

    my_seg8.stop_threads()
    pipeline.exit()
    del pipeline
    del my_seg8
    cv2.destroyAllWindows()
