#!/usr/bin/env python3

import copy
import threading
import time
import cv2
import numpy as np
import torch

from ultralytics import YOLO

# Set the current CUDA device using the index number of the GPU
torch.cuda.set_device(0)

class Seg8:
	'''
	Class for running YOLOV8-seg models efficiently (15 fps max on Jetson Nano 4GB).
	Based on the works of Ultralytics https://docs.ultralytics.com/tasks/segment/.
	'''
	def __init__(self, path_to_yolo):
		self.__class_names = ["Crimper", "Drill", "Hammer"]
		self.__class_weight = [0.486, 1.350, 0.930]	# [kg]
		self.__path_to_yolo = path_to_yolo

		self.__frame = None
		self.__depth_frame = None
		self.__new_frame_ready = False
		self.__new_result_ready = False
		self.__seg8_result = None
		self.__annotated_frame = None
		self.__annotated_frame_ready = False
		self.__obj_masks_indices = []
		self.__obj_masks_distance = []
		self.__stop = False

		# Threads for YOLO
		self.__yolo_thread = threading.Thread(target=self.__yolo_predict)
		self.__post_processing_thread = threading.Thread(target=self.__post_processing)
	# ---------------------------------------------------------------------------------------------------

	def __del__(self):
		'''
		Destructor to ensure threads are stopped and joined properly
		'''
		self.stop_threads()
		self.__stop = True
		self.__yolo_thread.join()
		self.__post_processing_thread.join()
	# ---------------------------------------------------------------------------------------------------

	def start_threads(self):
		'''
		Method to start YOLO prediction and post-processing threads to ensure efficient processing
		'''
		self.__frame = None
		self.__depth_frame = None
		self.__new_frame_ready = False
		self.__new_result_ready = False
		self.__seg8_result = None
		self.__annotated_frame = None
		self.__annotated_frame_ready = False
		self.__obj_masks_indices = []
		self.__obj_masks_distance = []
		self.__stop = False

		self.__yolo_thread.start()
		self.__post_processing_thread.start()
	# ---------------------------------------------------------------------------------------------------

	def stop_threads(self):
		'''
		Method to signal threads to stop
		'''
		self.__stop = True
	# ---------------------------------------------------------------------------------------------------

	def __yolo_predict(self):
		'''
		YOLO result prediction thread method
		'''
		model = YOLO(self.__path_to_yolo, verbose=False)
		
		# Wait until a frame is available or stop signal is received
		while self.__frame is None and not self.__stop:
			time.sleep(0.1)

		# Continuous prediction loop
		while not self.__stop:
			if self.__new_frame_ready:
				self.__new_frame_ready = False
				yolo_frame = copy.deepcopy(self.__frame)
				self.__seg8_result = model.predict(yolo_frame, verbose = False, max_det = 3, conf = 0.7, device = 0)[0]
				self.__new_result_ready = True
			else:
				time.sleep(0.002)
	# ---------------------------------------------------------------------------------------------------

	def __post_processing(self):
		'''
		Post-processing thread method.
		It creates annotated frames by drawing contours and masks, and calculates object distances using depth data.
		'''
		while not self.__stop: # Waiting loop, until there is a new result or stop signal is received
			if self.__new_result_ready:
				break
			else:
				time.sleep(0.01)

		# Create an empty mask for the full image size
		full_mask = np.zeros(self.__seg8_result.orig_img.shape[:2], dtype=np.uint8)

		while not self.__stop:
			if self.__new_result_ready:
				self.__new_result_ready = False
				img = np.copy(self.__seg8_result.orig_img)
				frame_to_annotate = np.copy(img)
				depth_frame_to_anotate = np.copy(self.__depth_frame)
				obj_masks_indices = []
				obj_masks_distance = []
				i = 0

				if self.__seg8_result.masks is not None:
					class_list = self.__seg8_result.boxes.cls.tolist()
					class_list = [int(cls) for cls in class_list]

					for mask in self.__seg8_result.masks:
						for xy in mask.xy:
							full_mask.fill(0)

							# Create contour mask 
							contour = xy.astype(np.int32).reshape(-1, 1, 2)
							_ = cv2.drawContours(full_mask, [contour], -1, (255), cv2.FILLED)

							mask_indices = np.argwhere(full_mask ==  255)

							# Assign colors based on class
							if class_list[i] == 0:
								mask_color = [0.5, 1, 0.5]
							elif class_list[i] == 1:
								mask_color = [1, 0.5, 0.5]
							else:
								mask_color = [0.5, 0.5, 1]

							frame_to_annotate[mask_indices[:,0], mask_indices[:,1], :] = img[mask_indices[:,0], mask_indices[:,1], :] * mask_color
							
							# Calculate the object's distance using the depth frame
							obj_distance = np.quantile(depth_frame_to_anotate[mask_indices[:,0], mask_indices[:,1]], 0.8)

							obj_masks_indices.append([mask_indices])
							obj_masks_distance.append([obj_distance])

							i = i + 1

				self.__annotated_frame = copy.deepcopy(frame_to_annotate)
				self.__obj_masks_indices = copy.deepcopy(obj_masks_indices)
				self.__obj_masks_distance = copy.deepcopy(obj_masks_distance)
				self.__annotated_frame_ready = True

			else:
				time.sleep(0.005)
	# ---------------------------------------------------------------------------------------------------

	def get_result_class_list(self):
		'''
		Method to retrieve the list of detected classes
		'''
		if self.__seg8_result is not None and self.__seg8_result.boxes is not None:
			return [int(cls) for cls in self.__seg8_result.boxes.cls.tolist()]
		else:
			return []
	# ---------------------------------------------------------------------------------------------------

	def get_class_names(self):
		'''
		Method to retrieve the class names
		'''
		return self.__class_names
	# ---------------------------------------------------------------------------------------------------

	def get_class_weight(self):
		'''
		Method to retrieve the class weights
		'''
		return self.__class_weight
	# ---------------------------------------------------------------------------------------------------

	def get_annotated_frame(self):
		'''
		Method to retrieve the annotated frame if ready
		'''
		if self.__annotated_frame_ready:
			self.__annotated_frame_ready = False
			return self.__annotated_frame
		else:
			return None
	# ---------------------------------------------------------------------------------------------------
	
	def get_obj_masks_indices(self):
		'''
		Method to retrieve object mask indices
		'''
		return self.__obj_masks_indices
	# ---------------------------------------------------------------------------------------------------

	def get_obj_masks_distance(self):
		'''
		Method to retrieve object mask distances
		'''
		return self.__obj_masks_distance
	# ---------------------------------------------------------------------------------------------------

	def update(self, frame, depth_frame):
		'''
		This function must be called in a loop
		'''
		# Update the current frame and depth frame
		self.__frame = frame
		self.__depth_frame = depth_frame

		# Signal that a new frame is ready
		if self.__frame is not None and self.__depth_frame is not None:
			self.__new_frame_ready = True
	# ---------------------------------------------------------------------------------------------------
