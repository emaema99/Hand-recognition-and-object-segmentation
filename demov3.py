#!/usr/bin/env python3

from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA, WINDOW_KEEPRATIO, putText, namedWindow, imshow, resizeWindow, waitKey, destroyAllWindows
from numpy import uint8, inf, min, nan, int32, isnan, ndarray, linalg, bool_, vstack, argmin, argwhere, mean, sum, full, copy, zeros
from time import time, sleep
from copy import deepcopy

from HandTrackerRendererV3 import HandTrackerRenderer
from CustomPipeline2 import CustomPipeline
from HandTrackerEdgeV4 import HandTracker
from seg8_nano_v5 import Seg8
from seg8_hand_utils_v2 import HostSpatialsCalc
from arduino_wifi_communication import ArduinoCommunicator

'''
Code for running YOLOV8 segmentation models on Jetson Nano 4GB and gesture recognition (hand tracking) on OAK devices.
Based on https://github.com/geaxgx/depthai_hand_tracker and on the works of Ultralytics https://docs.ultralytics.com/tasks/segment/.
'''

path_to_yolo = "/home/ema/Desktop/depthai-hand-segmentation/models/best_nano8.pt"
camera_fps = 15
num_elements_moving_avarage = 15
hand_obj_dist_threshold = 60 #[mm]
down_sampling = 5

DISPLAY = True
EXO_COMM = False

def add_grid_img(gesture_status, object_name, object_weight, fps, grid_height, grid_width, cell_height, cell_width):
    '''
    It creates a blank image grid with grasping status, object name, object weight, and FPS displayed.
    '''
    grid_img = zeros((grid_height, grid_width, 3), dtype=uint8)

    # Adds fixed text labels
    putText(grid_img, f'Status:', (10, cell_height // 2 + 5), FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, LINE_AA)
    putText(grid_img, f'Object:', (2*cell_width-30, cell_height // 2 + 5), FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, LINE_AA)
    putText(grid_img, f'Weight:', (10, cell_height + cell_height // 2), FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, LINE_AA)
    putText(grid_img, f'Fps:', (2*cell_width - 30, cell_height + cell_height // 2), FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, LINE_AA)

    # Adds dynamic data. Modify the blank spaces and font dimension to obtain a better visualization. 
    putText(grid_img, ' ' + gesture_status, (90, cell_height // 2 + 5), FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, LINE_AA)
    putText(grid_img, object_name, (2*cell_width + 60, cell_height // 2 + 5), FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, LINE_AA)
    putText(grid_img, ' ' + object_weight +' kg', (90, cell_height + cell_height // 2), FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, LINE_AA)
    putText(grid_img, f'{fps}', (2*cell_width + 60, cell_height + cell_height // 2), FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, LINE_AA)

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

    if time() - start_time > 1:
        fps = frame_count / (time() - start_time)
        start_time = time()
        frame_count = 0

    return start_time, frame_count, fps

def draw_mask(frame_to_annotate, mask, class_type):
    '''
    Mask appliance based on the class of the detected object. 
    Using a copy of the frame not to alter the original.
    '''
    img = copy(frame_to_annotate)
    annotated_frame = copy(frame_to_annotate)

    if class_type == 0:
        mask_color = [0.1, 0.1, 1]
    elif class_type == 1:
        mask_color = [1, 0.1, 0.1]
    else:
        mask_color = [0.1, 1, 0.1]

    annotated_frame[mask[:,0], mask[:,1], :] = img[mask[:,0], mask[:,1], :] * mask_color

    return annotated_frame

def calc_distances(masks_pixels, depth_frame, hand_spatial):
    if len(masks_pixels) > 0 or hand_spatial is not None:
        masks_spatial = []
        masks_pixels_in_range = []
        for i in range(len(masks_pixels)):
            # Calculate spatial data for each mask
            mask_spatial, mask_pixels_in_range = spatial_calc.calc_roi_each_point_spatials(depth_frame, masks_pixels[i], down_sampling)

            if isinstance(mask_spatial, ndarray):
                mask_spatial = mask_spatial.reshape(-1,3)

            masks_spatial.append(mask_spatial)
            masks_pixels_in_range.append(mask_pixels_in_range)

        # Process hand data (if available and if a hand was detected)
        if grasping_status and hand_spatial is not None:
            if len(masks_spatial) > 0:
                min_dist = inf
                min_dist_index = None
                min_index = None
                for i in range(len(masks_spatial)):
                    if masks_spatial[i] is not None and isinstance(masks_spatial[i], ndarray):
                        hand_spatial_matr = full(masks_spatial[i].shape, hand_spatial)
                        # Calculates the distance between the palm and the closest pixel of the mask.
                        dists = linalg.norm(hand_spatial_matr - masks_spatial[i], axis = 1)
                        mask_min_dist = min(dists)
                        if mask_min_dist < min_dist:
                            min_dist = deepcopy(mask_min_dist)
                            min_index = deepcopy(i)
                            mask_min_dist_index = argmin(dists)
                            min_dist_index = deepcopy(mask_min_dist_index)

                if min_index is not None and min_dist_index is not None:
                    return min_index, masks_spatial[min_index][min_dist_index,:], min_dist

    return None, [nan, nan, nan], None

if __name__ == '__main__':
    '''
    Initialize the pipeline (while setting the thresholds for the palm detection and landmark detection algorithms),
    the hand tracker (with frame size and padding parameters from the pipeline) and the hand tracker renderer
    '''
    pipeline = CustomPipeline(pd_score_thresh = 0.4, lm_score_thresh = 0.4, internal_fps = camera_fps)
    handTracker = HandTracker(pipeline.frame_size, pipeline.pad_h, pipeline.pad_w, pipeline.lm_score_thresh)
    handRender = HandTrackerRenderer(handTracker)
    depth_data = pipeline.getDepthData() # used to retrieve HFOV info
    spatial_calc = HostSpatialsCalc(pipeline.getDevice(), depth_data)

    # Initialize and start the YOLO segmentation model
    my_seg8 = Seg8(path_to_yolo, DISPLAY)
    my_seg8.start_threads()

    # Initialize variables for frame processing and FPS calculation
    start_time = time()
    frame_count = 0
    fps = 0

    # Initialize arrays and variables for grasping detection and object tracking
    grasping_status_arr = zeros(num_elements_moving_avarage, dtype=bool_)
    hand_spatial_arr = zeros([num_elements_moving_avarage, 3])
    obj_spatial_arr = zeros([num_elements_moving_avarage, 3])
    last_grasped_obj_name = "-"
    last_obj_weight = "-"
    released = True
    moving_avarage_index = 0

    # Initialize communication with the exosuit (if enabled).
    if EXO_COMM:
        communicator = ArduinoCommunicator(ip='192.168.0.21', port=50000)
        communicator.connect()

    try:
        # Capture depth and RGB frame from the camera pipeline.
        first_frame = pipeline.getFrame()
        first_depth_frame = pipeline.getDepthFrame() # UINT16 - mm

        while first_frame is None or first_depth_frame is None:
            first_frame = pipeline.getFrame()
            first_depth_frame = pipeline.getDepthFrame()
            sleep(0.1)

        # Sets the grid and cells dimensions, depending on the frame shape.
        grid_height = 100
        grid_width = first_frame.shape[1]
        grid_img = zeros((grid_height, grid_width, 3), dtype=uint8)
        cell_height = grid_height // 2
        cell_width = grid_width // 3
        grid_img = add_grid_img("Not Grasping", "-", "-", "0", grid_height, grid_width, cell_height, cell_width)
        combined_img = vstack((first_frame, grid_img))

        # Here the frame is resized for better visualization, while mantaining the ratio.
        if DISPLAY:
            namedWindow('Object segmentation & grasping detection', WINDOW_KEEPRATIO)
            imshow('Object segmentation & grasping detection', combined_img)
            resizeWindow('Object segmentation & grasping detection', 1280, 720)
            waitKey(1)

        # Save current frame as (i-2)th frame
        frame_i_2 = deepcopy(first_frame)
        depth_frame_i_2 = deepcopy(first_depth_frame)

        # Send the (i-2)th frame to the Seg8
        my_seg8.set_frame_to_seg(frame_i_2)
        handsData_i_2 = deepcopy(handTracker.getHandsData(pipeline.getManagerScriptOutput())) # frame (i-2)th

        # Take next frame and save it as (i-1)th frame
        frame_i_1 = deepcopy(pipeline.getFrame())
        depth_frame_i_1 = deepcopy(pipeline.getDepthFrame())

        # Wait for YOLOv8-seg to generate results
        result_i_2 = deepcopy(my_seg8.get_yolo_seg_result()) # blocking behaviour

        # Send (i-1)th frame to Seg8
        my_seg8.set_frame_to_seg(frame_i_1)
        handsData_i_1 = deepcopy(handTracker.getHandsData(pipeline.getManagerScriptOutput())) # frame (i-1)th

        # Start post processing on (i-2)th frame
        if DISPLAY:
            my_seg8.start_seg_post_processing(frame_i_2, result_i_2)
        # Preparing the post processing of the mask that will be ready at (i-2)th frame

        while True:
            # Capture depth and RGB (i)th frame from the camera pipeline
            frame_i = pipeline.getFrame() # blocking (15fps)
            depth_frame_i = pipeline.getDepthFrame()

            # Initialization of variables for frame annotation
            frame_final = frame_i_2
            obj_spatial = None
            hand = None
            class_names = None
            obj_weight = None
            grasping_status = False
            grasped_obj_name = "-"
            hand_obj_dist = "-"

            # Take results from Seg8 at (i-1)th frame
            result_i_1 = deepcopy(my_seg8.get_yolo_seg_result()) # blocking

            # Send (i)th frame to Seg8
            my_seg8.set_frame_to_seg(frame_i) # update (i)th frame, non-blocking

            # Perform hand tracker inference
            handsData_i = handTracker.getHandsData(pipeline.getManagerScriptOutput()) # (i)th frame

            if handsData_i_2:
                hand = handsData_i_2[0]
                grasping_status = hand.is_grasping
                hand_spatial = hand.xyz
                

            if DISPLAY:
                # Take (i-2)th post-processed frame
                annotated_frame_i_2, masks_indices_i_2, mask_contour_indices_i_2, inference_class_list_i_2 = deepcopy(my_seg8.get_seg_post_processing()) # blocking
                # Start post processing on (i-1)th frame
                my_seg8.start_seg_post_processing(frame_i_1, result_i_1)
            else:
                if result_i_1 is not None and result_i_1.boxes is not None:
                    inference_class_list_i_2 = [int(cls) for cls in result_i_1.boxes.cls.tolist()]
                else:
                    inference_class_list_i_2 = []
                mask_contour_indices_i_2 = []
                if result_i_1 is not None and result_i_1.masks is not None:
                    for mask in result_i_1.masks:
                        for xy in mask.xy:
                            contour = xy.astype(int32).reshape(-1, 1, 2)
                            contour_yx = zeros((contour.shape[0],2), dtype=int32)
                            contour_yx[:,0] = contour[:, 0, 1]
                            contour_yx[:,1] = contour[:, 0, 0]
                            mask_contour_indices_i_2.append([contour_yx])
                               

            # Calculates distance from the palm of the hand to the closest pixel of the object mask in (i-2)th frame
            if handsData_i_2 and hand_spatial is not None:
                if grasping_status:
                    if len(mask_contour_indices_i_2) > 0:
                        selected_index, selected_obj_spatial, selected_dist = calc_distances(mask_contour_indices_i_2, depth_frame_i_2, hand_spatial)
                        if selected_index is not None and selected_index >= 0 and selected_index < len(inference_class_list_i_2):
                            grasped_obj_name = my_seg8.get_class_names()[inference_class_list_i_2[selected_index]]
                            obj_weight = my_seg8.get_class_weight()[inference_class_list_i_2[selected_index]]
                            hand_obj_dist = int(selected_dist)

                            # Update moving average arrays with current data
                            hand_spatial_arr[moving_avarage_index, :] = hand_spatial
                            obj_spatial_arr[moving_avarage_index, :] = selected_obj_spatial

                            # Dealing with NaN values
                            hand_spatial_arr_without_nan_index = argwhere(~isnan(hand_spatial_arr)).reshape(-1)

                            if hand_spatial_arr_without_nan_index.shape[0] > 0:
                                hand_spatial_arr_without_nan = hand_spatial_arr[hand_spatial_arr_without_nan_index]
                                hand_spatial_avarage = mean(hand_spatial_arr_without_nan, axis=0)

                            obj_spatial_arr_without_nan_index = argwhere(~isnan(obj_spatial_arr)).reshape(-1)

                            if obj_spatial_arr_without_nan_index.shape[0] > 0:
                                obj_spatial_arr_without_nan = obj_spatial_arr[obj_spatial_arr_without_nan_index]
                                obj_spatial_avarage = mean(obj_spatial_arr_without_nan, axis=0)

                            # Calculates the 3D norm (spatial distance) between hand and object
                            hand_obj_dist = linalg.norm(hand_spatial_avarage - obj_spatial_avarage)
                            print("hand_obj_dist: ", hand_obj_dist)

                            if hand_obj_dist < hand_obj_dist_threshold:
                                grasping_status_arr[moving_avarage_index] = True
                            else:
                                grasping_status_arr[moving_avarage_index] = False

                            # Update moving average index
                            moving_avarage_index = (moving_avarage_index + 1) % num_elements_moving_avarage
                else:
                    # Handle case when the hand is not grasping
                    grasping_status_arr[moving_avarage_index] = False
                    hand_spatial_arr[moving_avarage_index, :] = [nan, nan, nan]
                    obj_spatial_arr[moving_avarage_index, :] = [nan, nan, nan]
                
                    # Update moving average index
                    moving_avarage_index = (moving_avarage_index + 1) % num_elements_moving_avarage

            start_time, frame_count, fps = calc_fps(start_time, frame_count, fps)

            

            # Calculate the moving average grasping status
            if sum(grasping_status_arr) > num_elements_moving_avarage/2:
                if released:
                    last_grasped_obj_name = grasped_obj_name
                    last_obj_weight = obj_weight
                    # If the communication is ON, sends the weight to the exosuit
                    if EXO_COMM:
                        communicator.send_weight(deepcopy(obj_weight))
                released = False
                grasping_status_avarage = True
            else:
                if EXO_COMM:
                    communicator.send_weight(0)
                released = True
                last_grasped_obj_name = "-"
                last_obj_weight = "-"
                grasping_status_avarage = False 
                
                
            if DISPLAY:
                # Draw hand annotations on the frame
                frame_final = handRender.draw(annotated_frame_i_2, handsData_i_2)
                
                # Update the grid image with current status
                grid_img = add_grid_img2(grasping_status_avarage, last_grasped_obj_name, last_obj_weight, fps, grid_height, grid_width, cell_height, cell_width)

                # Stack the main frame and the grid image vertically
                combined_img = vstack((frame_final, grid_img))

                namedWindow('Object segmentation & grasping detection', WINDOW_KEEPRATIO)
                imshow('Object segmentation & grasping detection', combined_img)
                resizeWindow('Object segmentation & grasping detection', 1280, 720)

                # Break the loop if 'q' key is pressed
                if waitKey(10) == ord('q'):
                    break
            else:
                print("fps: ", fps)


            # Update frames
            # frame (i-1) -> frame (i-2)
            frame_i_2 = deepcopy(frame_i_1)
            depth_frame_i_2 = deepcopy(depth_frame_i_1)
            handsData_i_2 = deepcopy(handsData_i_1)
            # frame (i) -> frame (i-1)
            frame_i_1 = deepcopy(frame_i)
            depth_frame_i_1 = deepcopy(depth_frame_i)
            handsData_i_1 = deepcopy(handsData_i)
            

    except KeyboardInterrupt:
        print("Interrupted by user")

    # Send weight = 0 to the communicator if EXO_COMM is enabled
    if EXO_COMM:
        communicator.send_weight(0)

    # Stop the segmentation threads, close pipeline and release resources
    my_seg8.stop_threads()
    pipeline.exit()
    del pipeline
    del my_seg8
    destroyAllWindows()
