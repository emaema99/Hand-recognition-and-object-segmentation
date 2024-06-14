#!/usr/bin/env python3

import math
import numpy as np
import depthai as dai

class HostSpatialsCalc:
    '''
    Class for calculating the depth.
    '''
    def __init__(self, device):
        # Initialize the HostSpatialsCalc class with device calibration data
        self.calibData = device.readCalibration()

        # Constants for depth processing
        self.DELTA = 5  # Take 10x10 depth pixels around point for depth averaging
        self.THRESH_LOW = 200 # 20 cm
        self.THRESH_HIGH = 1000 # 1.5 m
        self.HFOV = self.calibData.getFov(dai.CameraBoardSocket.CAM_A) # Horizontal field of view of the camera
    # --------------------------------------------------------------------------------------------

    def setLowerThreshold(self, threshold_low):
        self.THRESH_LOW = threshold_low
    # --------------------------------------------------------------------------------------------

    def setUpperThreshold(self, threshold_low):
        self.THRESH_HIGH = threshold_low
    # --------------------------------------------------------------------------------------------

    def setDeltaRoi(self, delta):
        '''Set the delta value for region of interest (ROI) size around a point'''
        self.DELTA = delta
    # --------------------------------------------------------------------------------------------

    def calc_point_spatial(self, depth_frame, point, margin=5):
        '''Calculate spatial coordinates for a given point in the depth frame.'''
        if margin > 0:
            x = min(max(point[0], margin), depth_frame.shape[1] - margin)
            y = min(max(point[1], margin), depth_frame.shape[0] - margin)
            roi = (x-margin, y-margin, x+margin, y+margin)
            spatials, centroid = self.calc_squared_roi_spatials(depth_frame, roi)
        else:
            spatials, centroid = self.calc_roi_spatials(depth_frame, point)
            pass

        return spatials, centroid
    # --------------------------------------------------------------------------------------------

    def calc_squared_roi_spatials(self, depth_frame, roi, averaging_method = np.mean):
        '''Calculate spatial coordinates for a squared region of interest (ROI) in the depth frame'''
        xmin, ymin, xmax, ymax = roi

        # Calculate the average depth in the ROI.
        depthROI = depth_frame[ymin:ymax, xmin:xmax]
        inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)

        averageDepth = averaging_method(depthROI[inRange])

        # Calculate the centroid of the ROI
        centroid = np.array([int((xmax + xmin) / 2), int((ymax + ymin) / 2)])

        # Calculate the middle of the depth image width and height
        midW = int(depth_frame.shape[1] / 2)
        midH = int(depth_frame.shape[0] / 2)
        bb_x_pos = centroid['x'] - midW
        bb_y_pos = centroid['y'] - midH

        # Calculate angles and spatial coordinates
        angle_x = self._calc_angle(depth_frame, bb_x_pos, self.HFOV)
        angle_y = self._calc_angle(depth_frame, bb_y_pos, self.HFOV)

        spatials = np.array([averageDepth, averageDepth * math.tan(angle_x), -averageDepth * math.tan(angle_y)])

        return spatials, centroid
    # --------------------------------------------------------------------------------------------

    def calc_roi_spatials(self, depth_frame, roi_pixels, averaging_method = np.mean):
        '''Calculate spatial coordinates for an arbitrary region of interest (ROI) defined by a set of pixels'''

        # Calculate the average depth in the ROI.
        if depth_frame is None or roi_pixels is None or len(roi_pixels)==0:
            return None, None

        # Copy depth frame and ROI pixels to avoid modifying original data
        depth_frame_c = np.copy(depth_frame)
        roi_pixels_c = np.copy(np.array(roi_pixels).reshape(-1,2))

        # Extract depth values at the ROI pixels
        roi_depth_values = depth_frame_c[roi_pixels_c[:,0], roi_pixels_c[:,1]]

        # Remove invalid and out-of-range depth values
        roi_depth_values_valid, roi_pixels_in_range_valid = self.remove_nan_and_out_of_range_points(roi_depth_values, roi_pixels_c)

        if roi_depth_values_valid is None or roi_pixels_in_range_valid is None:
            return None, None

        # Calculate the average depth of the valid ROI pixels
        averageDepth = averaging_method(roi_depth_values_valid)

        # Calculate the centroid of the valid ROI pixels
        centroid = np.array([int(np.sum(roi_pixels_in_range_valid[:,0]) / len(roi_pixels_in_range_valid[:,0])),
                             int(np.sum(roi_pixels_in_range_valid[:,1]) / len(roi_pixels_in_range_valid[:,1]))])

        # Calculate the middle of the depth image width and height
        midW = int(depth_frame.shape[1] / 2)
        midH = int(depth_frame.shape[0] / 2)
        bb_x_pos = centroid[0] - midW
        bb_y_pos = centroid[1] - midH
        print("x_pos_arr x: ", bb_x_pos)
        print("y_pos_arr y: ", bb_y_pos)

        # Calculate angles and spatial coordinates
        angle_x = self._calc_angle(depth_frame, bb_x_pos, self.HFOV)
        angle_y = self._calc_angle(depth_frame, bb_y_pos, self.HFOV)

        spatials = np.array([averageDepth * math.tan(angle_x), -averageDepth * math.tan(angle_y), averageDepth])

        return spatials, centroid
    # --------------------------------------------------------------------------------------------

    def calc_roi_each_point_spatials(self, depth_frame, roi_pixels, down_sampling=1, averaging_method = np.mean):
        '''Calculate spatial coordinates for each point in an ROI with optional down-sampling'''

        if depth_frame is None or roi_pixels is None or len(roi_pixels)==0:
            return None, None

        # Copy depth frame and ROI pixels to avoid modifying original data
        depth_frame_c = np.copy(depth_frame)
        roi_pixels_c = np.copy(np.array(roi_pixels).reshape(-1,2))

        # Extract depth values at the ROI pixels
        roi_depth_values = depth_frame_c[roi_pixels_c[:,0], roi_pixels_c[:,1]]

        # Remove background noise and outliers
        obj_depth = np.quantile(roi_depth_values, 0.7)
        roi_depth_values2_index = np.argwhere(roi_depth_values < obj_depth + 5).reshape(-1)

        if roi_depth_values2_index.size < 1:
            return None, None

        roi_depth_values2 = roi_depth_values[roi_depth_values2_index]
        roi_pixels_2 = roi_pixels_c[roi_depth_values2_index,:]
        roi_depth_values2_index = np.argwhere(roi_depth_values2 > obj_depth - 5).reshape(-1)

        if roi_depth_values2_index.size < 1:
            return None, None

        roi_depth_values = roi_depth_values2[roi_depth_values2_index]
        roi_pixels_c = roi_pixels_2[roi_depth_values2_index,:]

        # Remove invalid and out-of-range depth values
        roi_depth_values_valid, roi_pixels_in_range_valid = self.remove_nan_and_out_of_range_points(roi_depth_values, roi_pixels_c)

        if roi_depth_values_valid is None or roi_pixels_in_range_valid is None:
            return None, None

        # Down-sample the valid ROI pixels if specified
        roi_depth_values_downsampled = roi_depth_values_valid[::down_sampling]
        roi_pixels_in_range_downsampled = roi_pixels_in_range_valid[::down_sampling,:]

        if not roi_depth_values_downsampled.size > 5:
            return None, None

        # Calculate the middle of the depth image width and height
        midW = int(depth_frame_c.shape[1] / 2)
        midH = int(depth_frame_c.shape[0] / 2)

        # Calculate positions relative to the center of the image
        x_pos_arr = roi_pixels_in_range_downsampled[:,0] - midW
        y_pos_arr = roi_pixels_in_range_downsampled[:,1] - midH

        # Calculate angles for each ROI pixel
        x_angle_tan_arr = np.zeros(len(x_pos_arr))
        y_angle_tan_arr = np.zeros(len(y_pos_arr))
        
        for i in range(len(y_angle_tan_arr)):
            x_angle_tan_arr[i] = math.tan(self._calc_angle(depth_frame, x_pos_arr[i], self.HFOV))
            y_angle_tan_arr[i] = math.tan(self._calc_angle(depth_frame, y_pos_arr[i], self.HFOV))

        # Calculate spatial coordinates for each ROI pixel
        spatials = np.zeros([len(x_angle_tan_arr), 3])
        spatials[:,0] = roi_depth_values_downsampled * x_angle_tan_arr
        spatials[:,1] = (roi_depth_values_downsampled * y_angle_tan_arr) * (-1)
        spatials[:,2] = roi_depth_values_downsampled

        return spatials
    # --------------------------------------------------------------------------------------------

    def _calc_angle(self, frame, offset, HFOV):
        '''Calculate the angle for a given offset and horizontal field of view'''
        return math.atan(math.tan(HFOV / 2.0) * offset / (frame.shape[1] / 2.0))
    # --------------------------------------------------------------------------------------------

    def remove_nan_and_out_of_range_points(self, roi_depth_values, roi_pixels):
        '''Remove NaN and out-of-range depth values from ROI pixels'''
        indices_not_nan = np.argwhere(~np.isnan(roi_depth_values)).reshape(-1)
        if indices_not_nan.shape[0] < 1:
            return None, None
        roi_depth_values_not_nan = roi_depth_values[indices_not_nan]
        roi_pixels_not_nan = roi_pixels[indices_not_nan,:]

        indices_in_range_min = np.argwhere(roi_depth_values_not_nan > self.THRESH_LOW).reshape(-1)
        if indices_in_range_min.shape[0] < 1:
            return None, None
        roi_depth_values_in_range_min = roi_depth_values_not_nan[indices_in_range_min]
        roi_pixels_in_range_min = roi_pixels_not_nan[indices_in_range_min,:]

        indices_in_range = np.argwhere(roi_depth_values_in_range_min < self.THRESH_HIGH).reshape(-1)
        if indices_in_range.shape[0] < 1:
            return None, None
        roi_depth_values_valid = roi_depth_values_in_range_min[indices_in_range]
        roi_pixels_in_range_valid = roi_pixels_in_range_min[indices_in_range,:]

        return roi_depth_values_valid, roi_pixels_in_range_valid
    # --------------------------------------------------------------------------------------------
