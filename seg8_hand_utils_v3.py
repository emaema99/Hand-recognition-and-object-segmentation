#!/usr/bin/env python3

from math import tan, atan
from numpy import deg2rad, isnan, array, copy, sum, zeros, argwhere, mean
from depthai import CameraBoardSocket

class HostSpatialsCalc:
    '''
    Class for calculating the spatial coordinates for a ROI.
    Adapted from https://github.com/luxonis/depthai-experiments/blob/master/gen2-calc-spatials-on-host/calc.py
    '''
    def __init__(self, device, depth_data):
        '''
        Required information for calculating spatial coordinates on the host. FOV of color camera is measured for 
        the full resolution, 4056x3040, so it would need to scale back accordingly if 4k/1080p is selected.
        4k is 3840x2160 central crop of 4056x3040, so there is a HFOV loss -> HFOV will be scaled back by 3840/4056.
        1080p has the same FOV as 4k (it's 4k downscaled/binned).
        '''
        self.calibData = device.readCalibration() # Initialize the class with device calibration data
        self.HFOV = deg2rad(self.calibData.getFov(CameraBoardSocket(depth_data.getInstanceNum()), useSpec=False))

        # Constants for depth processing
        self.DELTA = 5  # Take 10x10 depth pixels around point for depth averaging
        self.THRESH_LOW = 200 # 20 cm
        self.THRESH_HIGH = 2000 # 2 m
    # --------------------------------------------------------------------------------------------

    def setDeltaRoi(self, delta):
        '''
        Set the delta value for region of interest (ROI) size around a point.
        '''
        self.DELTA = delta
    # --------------------------------------------------------------------------------------------

    def calc_point_spatial(self, depth_frame, point, margin=5):
        '''
        Calculate spatial coordinates for a given point in the depth frame.
        '''
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

    def calc_squared_roi_spatials(self, depth_frame, roi, averaging_method = mean):
        '''
        Calculate spatial coordinates for a squared region of interest (ROI) in the depth frame.
        '''
        xmin, ymin, xmax, ymax = roi

        # Calculate the average depth in the ROI.
        depthROI = depth_frame[ymin:ymax, xmin:xmax]
        inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)

        averageDepth = averaging_method(depthROI[inRange])

        # Calculate the centroid of the ROI
        centroid = array([int((xmax + xmin) / 2), int((ymax + ymin) / 2)])

        # Calculate the middle of the depth image width and height
        midW = int(depth_frame.shape[1] / 2)
        midH = int(depth_frame.shape[0] / 2)
        bb_x_pos = centroid['x'] - midW
        bb_y_pos = centroid['y'] - midH

        # Calculate angles and spatial coordinates
        angle_x = self._calc_angle(depth_frame, bb_x_pos)
        angle_y = self._calc_angle(depth_frame, bb_y_pos)

        # spatials = array([averageDepth, averageDepth * tan(angle_x), -averageDepth * tan(angle_y)])
        spatials = array([averageDepth * tan(angle_x), -averageDepth * tan(angle_y), averageDepth])

        return spatials, centroid
    # --------------------------------------------------------------------------------------------

    def calc_roi_spatials(self, depth_frame, roi_pixels, averaging_method = mean):
        '''
        Calculate spatial coordinates for an arbitrary region of interest (ROI) defined by a set of pixels in the depth frame.
        '''
        # Check if data is there
        if depth_frame is None or roi_pixels is None or len(roi_pixels)==0:
            return None, None

        # Copy depth frame and ROI pixels to avoid modifying original data
        depth_frame_c = copy(depth_frame)
        roi_pixels_c = copy(array(roi_pixels).reshape(-1,2))

        # Extract depth values at the ROI pixels
        roi_depth_values = depth_frame_c[roi_pixels_c[:,0], roi_pixels_c[:,1]]

        # Remove invalid and out-of-range depth values
        roi_depth_values_valid, roi_pixels_in_range_valid = self.remove_nan_and_out_of_range_points(roi_depth_values, roi_pixels_c)

        if roi_depth_values_valid is None or roi_pixels_in_range_valid is None:
            return None, None

        # Calculate the average depth of the valid ROI pixels
        averageDepth = averaging_method(roi_depth_values_valid)

        # Calculate the centroid of the valid ROI pixels
        centroid = array([int(sum(roi_pixels_in_range_valid[:,1]) / len(roi_pixels_in_range_valid[:,1])), #centroid[0]->pixel y, spatial x
                             int(sum(roi_pixels_in_range_valid[:,0]) / len(roi_pixels_in_range_valid[:,0]))])#centroid[1]->pixel x, spatial y

        # Calculate the middle of the depth image width and height
        midW = int(depth_frame.shape[1] / 2) # colonne, spatial x
        midH = int(depth_frame.shape[0] / 2) # righe, spatial y
        bb_x_pos = centroid[0] - midW
        bb_y_pos = centroid[1] - midH

        # Calculate angles and spatial coordinates
        angle_x = self._calc_angle(depth_frame, bb_x_pos, self.HFOV)
        angle_y = self._calc_angle(depth_frame, bb_y_pos, self.HFOV)

        # Spatials = x,y,z info
        spatials = array([averageDepth * tan(angle_x), -averageDepth * tan(angle_y), averageDepth])

        return spatials, centroid
    # --------------------------------------------------------------------------------------------

    def calc_roi_each_point_spatials(self, depth_frame, roi_pixels, down_sampling=1):
        '''
        Calculate spatial coordinates for each point in an ROI with optional down-sampling
        '''
        if depth_frame is None or roi_pixels is None or len(roi_pixels) == 0:
            print("Warning: Invalid input to calc_roi_each_point_spatials")
            return None, None

        # Copy depth frame
        depth_frame_c = copy(depth_frame)

        # Handle roi_pixels reshaping
        try:
            roi_pixels_c = copy(array(roi_pixels).reshape(-1, 2))
        except ValueError as e:
            print(f"Error reshaping roi_pixels: {e}")
            print(f"roi_pixels shape: {array(roi_pixels).shape}")
            print(f"roi_pixels content: {roi_pixels}")
            return None, None

        # Extract depth values at the ROI pixels
        try:
            roi_depth_values = depth_frame_c[roi_pixels_c[:, 0], roi_pixels_c[:, 1]]
        except IndexError as e:
            print(f"Error extracting depth values: {e}")
            print(f"depth_frame shape: {depth_frame_c.shape}")
            print(f"roi_pixels_c shape: {roi_pixels_c.shape}")
            return None, None

        # Remove invalid and out-of-range depth values
        roi_depth_values_valid, roi_pixels_in_range_valid = self.remove_nan_and_out_of_range_points(roi_depth_values, roi_pixels_c)
        if roi_depth_values_valid is None or roi_pixels_in_range_valid is None:
            print("Warning: No valid depth values after filtering")
            return None, None

        # Down-sample the valid ROI pixels if specified
        roi_depth_values_downsampled = roi_depth_values_valid[::down_sampling]
        roi_pixels_in_range_downsampled = roi_pixels_in_range_valid[::down_sampling, :]
        if not roi_depth_values_downsampled.size > 5:
            print("Warning: Not enough valid points after down-sampling")
            return None, None

        # Calculate the middle of the depth image width and height
        midW = int(depth_frame_c.shape[1] / 2)
        midH = int(depth_frame_c.shape[0] / 2)

        # Calculate positions relative to the center of the image
        x_pos_arr = roi_pixels_in_range_downsampled[:, 1] - midW
        y_pos_arr = roi_pixels_in_range_downsampled[:, 0] - midH

        # Calculate angles for each ROI pixel
        value = tan(self.HFOV / 2.0) / (depth_frame.shape[1] / 2.0)
        x_angle_tan_arr = x_pos_arr * value
        y_angle_tan_arr = y_pos_arr * value

        # Calculate spatial coordinates for each ROI pixel
        spatials = zeros([len(x_angle_tan_arr), 3])
        spatials[:, 0] = roi_depth_values_downsampled * x_angle_tan_arr
        spatials[:, 1] = (roi_depth_values_downsampled * y_angle_tan_arr) * (-1)
        spatials[:, 2] = roi_depth_values_downsampled

        return spatials, roi_pixels_in_range_downsampled
    # --------------------------------------------------------------------------------------------

    def _calc_angle(self, frame, offset):
        '''
        Calculate the angle for a given offset and horizontal field of view
        '''
        return atan(tan(self.HFOV / 2.0) * offset / (frame.shape[1] / 2.0))
    # --------------------------------------------------------------------------------------------

    def remove_nan_and_out_of_range_points(self, roi_depth_values, roi_pixels):
        '''
        Remove NaN and out-of-range depth values from ROI pixels.
        '''
        indices_not_nan = argwhere(~isnan(roi_depth_values)).reshape(-1)
        if indices_not_nan.shape[0] < 1:
            return None, None
        roi_depth_values_not_nan = roi_depth_values[indices_not_nan]
        roi_pixels_not_nan = roi_pixels[indices_not_nan,:]

        indices_in_range_min = argwhere(roi_depth_values_not_nan > self.THRESH_LOW).reshape(-1)
        if indices_in_range_min.shape[0] < 1:
            return None, None
        roi_depth_values_in_range_min = roi_depth_values_not_nan[indices_in_range_min]
        roi_pixels_in_range_min = roi_pixels_not_nan[indices_in_range_min,:]

        indices_in_range = argwhere(roi_depth_values_in_range_min < self.THRESH_HIGH).reshape(-1)
        if indices_in_range.shape[0] < 1:
            return None, None
        roi_depth_values_valid = roi_depth_values_in_range_min[indices_in_range]
        roi_pixels_in_range_valid = roi_pixels_in_range_min[indices_in_range,:]

        return roi_depth_values_valid, roi_pixels_in_range_valid
    # --------------------------------------------------------------------------------------------
