#!/usr/bin/env python3

from numpy import copy, zeros, argwhere, uint8, int32
from cv2 import drawContours, FILLED
from time import sleep
from copy import deepcopy
from torch import cuda
from threading import Thread
from ultralytics import YOLO

# Set the current CUDA device using the index number of the GPU
cuda.set_device(0)

class Seg8:
	'''
	Class for running YOLOV8-seg models efficiently (15 fps max on Jetson Nano 4GB).
	Based on the works of Ultralytics https://docs.ultralytics.com/tasks/segment/.
	'''
	def __init__(self, path_to_yolo, post_processing_active):
		self.__class_names = ["Crimper", "Drill", "Hammer"]
		self.__class_weight = [0.486, 1.350, 0.930]	# [kg]
		self.__path_to_yolo = path_to_yolo
		self.__post_processing_active = post_processing_active

		self.__frame = None
		self.__depth_frame = None
		self.__new_frame_ready = False
		self.__new_result_ready = False
		self.__seg8_result = None
		self.__start_post_processing = False
		self.__frame_to_post_proc = None
		self.__result_to_post_proc = None
		self.__annotated_frame = None
		self.__annotated_frame_ready = False
		self.__obj_masks_indices = []
		self.__obj_masks_contour_indices = []
		self.__inference_class_list = []
		self.__stop = False

		# Threads for YOLO
		self.__yolo_thread = Thread(target = self.__yolo_predict)
		if self.__post_processing_active:
			self.__post_processing_thread = Thread(target = self.__post_processing)
	# ---------------------------------------------------------------------------------------------------

	def __del__(self):
		'''
		Destructor to ensure threads are stopped and joined properly
		'''
		self.stop_threads()
		self.__stop = True
		self.__yolo_thread.join()
		if self.__post_processing_active:
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
		self.__start_post_processing = False
		self.__frame_to_post_proc = None
		self.__result_to_post_proc = None
		self.__annotated_frame = None
		self.__annotated_frame_ready = False
		self.__obj_masks_indices = []
		self.__obj_masks_contour_indices = []
		self.__inference_class_list = []
		self.__stop = False

		self.__yolo_thread.start()
		if self.__post_processing_active:
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
			sleep(0.1)

		# Continuous prediction loop
		while not self.__stop:
			if self.__new_frame_ready:
				self.__new_frame_ready = False
				# yolo_frame = deepcopy(self.__frame)
				self.__seg8_result = model.predict(self.__frame, verbose = False, max_det = 3, conf = 0.7, device = 0)[0]
				self.__new_result_ready = True
			else:
				sleep(0.002)
	# ---------------------------------------------------------------------------------------------------

	def get_yolo_seg_result(self):
		while not self.__new_result_ready and not self.__stop:
			sleep(0.002)
		self.__new_result_ready = False
		return self.__seg8_result
	# ---------------------------------------------------------------------------------------------------

	def set_frame_to_seg(self, frame):
		self.__frame = deepcopy(frame)
		self.__new_frame_ready = True
	# ---------------------------------------------------------------------------------------------------

	def start_seg_post_processing(self, frame_to_post_proc, result_to_post_proc):
		self.__frame_to_post_proc = deepcopy(frame_to_post_proc)
		self.__result_to_post_proc = deepcopy(result_to_post_proc)
		self.__start_post_processing = True
	# ---------------------------------------------------------------------------------------------------

	def get_seg_post_processing(self):
		while not self.__annotated_frame_ready and not self.__stop:
			sleep(0.002)
		self.__annotated_frame_ready = False
		return self.__annotated_frame, self.__obj_masks_indices, self.__obj_masks_contour_indices, self.__inference_class_list
	# ---------------------------------------------------------------------------------------------------

	def __post_processing(self):
		'''
		Post-processing thread method.
		It creates annotated frames by drawing contours and masks, and calculates object distances using depth data.
		'''
		while not self.__stop: # Waiting loop, until there is a new result or stop signal is received
			if self.__start_post_processing:
				break
			else:
				sleep(0.01)

		# Create an empty mask for the full image size
		full_mask = zeros(self.__result_to_post_proc.orig_img.shape[:2], dtype=uint8)

		while not self.__stop:
			if self.__start_post_processing:
				self.__start_post_processing = False
				# if self.__frame_to_post_proc is not None:
				img = copy(self.__frame_to_post_proc)
				frame_to_annotate = copy(img)
				obj_masks_indices = []
				obj_masks_contour_indices = []
				i = 0

				if self.__result_to_post_proc.masks is not None:
					class_list = self.__result_to_post_proc.boxes.cls.tolist()
					class_list = [int(cls) for cls in class_list]

					for mask in self.__result_to_post_proc.masks:
						for xy in mask.xy:
							full_mask.fill(0)

							# Create contour mask 
							contour = xy.astype(int32).reshape(-1, 1, 2)
							_ = drawContours(full_mask, [contour], -1, (255), FILLED)

							mask_indices = argwhere(full_mask ==  255)

							# Assign colors based on class
							if class_list[i] == 0:
								mask_color = [0.5, 1, 0.5]
							elif class_list[i] == 1:
								mask_color = [1, 0.5, 0.5]
							else:
								mask_color = [0.5, 0.5, 1]

							frame_to_annotate[mask_indices[:,0], mask_indices[:,1], :] = img[mask_indices[:,0], mask_indices[:,1], :] * mask_color
							
							contour_yx = zeros((contour.shape[0],2), dtype=int32)
							contour_yx[:,0] = contour[:, 0, 1]
							contour_yx[:,1] = contour[:, 0, 0]

							obj_masks_indices.append([mask_indices])
							obj_masks_contour_indices.append([contour_yx])

							i = i + 1

				self.__annotated_frame = frame_to_annotate	# dovrebbe essereci una deepcopy
				self.__obj_masks_indices = obj_masks_indices	# dovrebbe essereci una deepcopy
				self.__obj_masks_contour_indices = obj_masks_contour_indices	# dovrebbe essereci una deepcopy

				if self.__result_to_post_proc is not None and self.__result_to_post_proc.boxes is not None:
					self.__inference_class_list = [int(cls) for cls in self.__result_to_post_proc.boxes.cls.tolist()]
				else:
					self.__inference_class_list = []

				self.__annotated_frame_ready = True

			else:
				sleep(0.005)
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
		
	def get_obj_masks_contour_indices(self):
		'''
		Method to retrieve object mask indices
		'''
		return self.__obj_masks_contour_indices
	# ---------------------------------------------------------------------------------------------------
