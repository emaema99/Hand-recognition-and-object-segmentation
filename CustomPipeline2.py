#!/usr/bin/env python3

import numpy as np
import mediapipe_utils as mpu
import depthai as dai
import sys
import re

from pathlib import Path
from collections import namedtuple
from numpy.lib.arraysetops import isin
from string import Template


class CustomPipeline:
	"""
    Custom Pipeline for executing Mediapipe (https://pypi.org/project/mediapipe/) for hand spatial tracking.
    This class derive from a public github repository (https://github.com/geaxgx/depthai_hand_tracker).
    The default parameter of this class is to detect one hand per frame.
    We assume to use the OAK-D Lite camera, hence that the depth information is available as default.
    
    >>ARGUMENTS:
    - pd_score: 			confidence score to determine whether a detection is reliable (a float between 0 and 1).
    - pd_nms_thresh: 		NMS threshold. 
    - lm_score_thresh : 	confidence score to determine whether landmarks prediction is reliable (a float between 0 and 1).
    - use_world_landmarks: 	boolean. The landmarks model yields 2 types of 3D coordinates : 
                    		- coordinates expressed in pixels in the image, always stored in hand.landmarks,
                    		- coordinates expressed in meters in the world, stored in hand.world_landmarks.
    - xyz : 				boolean, when True calculate the (x, y, z) coords of the detected palms.
    - crop : 				boolean which indicates if square cropping on source images is applied or not
    						Default Set to True for matching the square imagesize input of Yolo
    - internal_fps : 		when using the internal color camera as input source, set its FPS to this value (calling setFps()).
    - resolution : 			sensor resolution "full" (1920x1080) or "ultra" (3840x2160),
    - internal_frame_height : when using the internal color camera, set the frame height (calling setIspScale()).
                    		The width is calculated accordingly to height and depends on value of 'crop'.
    - use_gesture : 		boolean, when True, recognize hand poses froma predefined set of poses
                    		(ONE, TWO, THREE, FOUR, FIVE, OK, PEACE, FIST)
    - use_handedness_average : boolean, when True the handedness is the average of the last collected handednesses.
							This brings robustness since the inferred robustness is not reliable on ambiguous hand poses.
							When False, handedness is the last inferred handedness.
    - single_hand_tolerance_thresh (Duo mode only) : In Duo mode, if there is only one hand in a frame, 
							in order to know when a second hand will appear you need to run the palm detection 
							in the following frames. Because palm detection is slow, you may want to delay 
							the next time you will run it. 'single_hand_tolerance_thresh' is the number of 
							frames during only one hand is detected before palm detection is run again.   
    - lm_nb_threads : 		1 or 2 (default=2), number of inference threads for the landmark model
    - use_same_image (Edge Duo mode only) : boolean, when True, use the same image when inferring the landmarks of the 2 hands
							(setReusePreviousImage(True) in the ImageManip node before the landmark model). 
							When True, the FPS is significantly higher but the skeleton may appear shifted on one of the 2 hands.
    - stats : 				boolean, when True, display some statistics when exiting.   
    - trace : 				int, 0 = no trace, otherwise print some debug messages or show output of ImageManip nodes
							if trace & 1, print application level info like number of palm detections,
							if trace & 2, print lower level info like when a message is sent or received by the manager script node,
							if trace & 4, show in cv2 windows outputs of ImageManip node,
							if trace & 8, save in file tmp_code.py the python code of the manager script node
							Ex: if trace==3, both application and low level info are displayed.

	>> ATTRIBUTES:
	- def createPipeline(self): 		Define and configure all the Pipeline's Nodes
	- def build_manager_script(self): 	Is called inside createPipeline and return the node Manager Script
	- def getNeuralNetwork(self): 		Return the path for all the .blob file and the Manager Script.py to create the pipeline
	- def exit(self):					Stop the pipeline closing the device
    """

	def __init__(self,
                pd_score_thresh = 0.9, 
                pd_nms_thresh = 0.3, 
                lm_model = "lite", 
                lm_score_thresh = 0.4, 
                use_world_landmarks = True,
                xyz = True,
                crop = False, 
                internal_fps = 15, 
                resolution = "full", 
                internal_frame_height = 288,
                use_gesture = True, 
                use_handedness_average = True, 
                single_hand_tolerance_thresh = 10, 
                use_same_image = True,
                lm_nb_threads = 1, # in [1, 2] == numero di mani. marco aveva messo 2
                stats = False,
                trace = 0 
                ):

		self.palm_detection_model, self.landmark_model, self.post_process_palm_detection_model, self.manager_script_solo = self.getNeuralNetwork()
  
		assert lm_nb_threads in [1, 2]
		self.lm_nb_threads = lm_nb_threads

		self.pd_score_thresh = pd_score_thresh
		self.pd_nms_thresh = pd_nms_thresh
		self.lm_score_thresh = lm_score_thresh

		self.syncNN = True
		self.xyz = xyz
		self.crop = crop 
		self.use_world_landmarks = use_world_landmarks
		   
		self.trace = trace
		self.use_gesture = use_gesture
		self.use_handedness_average = use_handedness_average
		self.single_hand_tolerance_thresh = single_hand_tolerance_thresh
		self.use_same_image = use_same_image

		# Device Object Declaration
		self.device = dai.Device()

		# ==== Setting Camera Parameters ====
		# Check USB Speed
		usb_speed = self.device.getUsbSpeed()
		print(f"OAK Connection USB speed: {str(usb_speed).split('.')[-1]}")

		# Defining OAK internal color camera as source video
		self.input_type = "rgb"

		# Setting Defaul Resolution
		self.resolution = (1920, 1080)
		print("Sensor resolution:", self.resolution)

		# Check if the device supports stereo
		cameras = self.device.getConnectedCameras()
		if not(dai.CameraBoardSocket.CAM_B in cameras and dai.CameraBoardSocket.CAM_C in cameras):
			print("FATAL ERROR: depth unavailable on this device, ADIOS")
			sys.exit(1)

		# Set Camera FPS for lm_model = lite configuration
		self.internal_fps =  internal_fps
		print(f"Internal camera FPS set to: {self.internal_fps}")

		# Used when saving the output in a video file. Should be close to the real fps
		self.video_fps = self.internal_fps 

		# Define Video and Image Size
		# The internal_frame_height must be set according to input image size of YOLO
		if self.crop:
			self.frame_size, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height, self.resolution)
			self.img_h = self.img_w = self.frame_size
			self.pad_w = self.pad_h = 0
			self.crop_w = (int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1])) - self.img_w) // 2
		else: # with crop = False we should have a rectangular 512x288 preview
			width, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height * self.resolution[0] / self.resolution[1], self.resolution, is_height=False)
			self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
			self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
			self.pad_h = (self.img_w - self.img_h) // 2
			self.pad_w = 0
			self.frame_size = self.img_w
			self.crop_w = 0
		print(f"Internal camera image size: {self.img_w} x {self.img_h} - pad_h: {self.pad_h}")

		# ==== Define and start pipeline ====
		#Pipeline Creation
		self.device.startPipeline(self.createPipeline())

		# Define data queues 
		self.cam_video_preview = self.device.getOutputQueue(name = "RGB preview", maxSize=1, blocking=False)
		self.stereo_queue = self.device.getOutputQueue(name = "stereo", maxSize=1, blocking=False)
		self.manager_out = self.device.getOutputQueue(name = "manager_out", maxSize=1, blocking=False)
    # ------------------------------------------------------------------------------------------------------------------

	def createPipeline(self):
		print("Creating pipeline...")

		# Start defining a pipeline
		pipeline = dai.Pipeline()
		pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)
		self.pd_input_length = 128

		# ==== Manager Script Node ====
		manager_script = pipeline.create(dai.node.Script)
		manager_script.setScript(self.build_manager_script()) #Create the Manager Node

		# ==== Camera RGB ====
		print("Creating Color Camera...")
		cam = pipeline.createColorCamera()

		# Properties
		cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
		cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
		cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
		cam.setInterleaved(False)
		cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
		cam.setFps(self.internal_fps)
		cam.setVideoSize(self.img_w, self.img_h)
		cam.setPreviewSize(self.img_w, self.img_h)

		# ==== Stereo and Depth Camera ====
		print("Creating MonoCameras, Stereo and SpatialLocationCalculator nodes...")
		# For now, RGB needs fixed focus to properly align with depth
		calib_data = self.device.readCalibration()
		calib_lens_pos = calib_data.getLensPosition(dai.CameraBoardSocket.CAM_A)
		print(f"RGB calibration lens position: {calib_lens_pos}")
		cam.initialControl.setManualFocus(calib_lens_pos)

		left = pipeline.createMonoCamera()
		right = pipeline.createMonoCamera()
		stereo = pipeline.createStereoDepth()

		# Properties
		left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
		left.setCamera("left")
		left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
		left.setFps(self.internal_fps)

		right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
		right.setCamera("right")
		right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
		right.setFps(self.internal_fps)

		stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
		stereo.setConfidenceThreshold(150) #230
		stereo.setLeftRightCheck(True) # LR-check is required for depth alignment
		stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
		stereo.setSubpixel(False)  # subpixel True brings latency

		# === Spatial Location Calculator ====
		spatial_location_calculator = pipeline.createSpatialLocationCalculator()

		# Properties
		spatial_location_calculator.setWaitForConfigInput(True)
		spatial_location_calculator.inputDepth.setBlocking(False)
		spatial_location_calculator.inputDepth.setQueueSize(1)

		# ==== Pre-processing Palm Detection Image Manipulation ====
		# Define palm detection pre processing: resize preview to (self.pd_input_length, self.pd_input_length)
		print("Creating Palm Detection pre processing image manip...")
		pre_pd_manip = pipeline.create(dai.node.ImageManip)

		# Properties 
		pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3) # palm det input length = 128
		pre_pd_manip.setWaitForConfigInput(True)
		pre_pd_manip.inputImage.setQueueSize(1)
		pre_pd_manip.inputImage.setBlocking(False)

		# ==== Palm Detection ====
		print("Creating Palm Detection Neural Network...")
		pd_nn = pipeline.create(dai.node.NeuralNetwork)

		# Properties
		pd_nn.setBlobPath(self.palm_detection_model)

		# ==== Post-processing Palm Detection ====
		print("Creating Palm Detection post processing Neural Network...")
		post_pd_nn = pipeline.create(dai.node.NeuralNetwork)

		# Properties
		post_pd_nn.setBlobPath(self.post_process_palm_detection_model)


		# ==== Pre-processing Landmark Image Manipulation ====
		print("Creating Hand Landmark pre processing image manip...") 
		pre_lm_manip = pipeline.create(dai.node.ImageManip)

		# Properties
		self.lm_input_length = 224
		pre_lm_manip.setMaxOutputFrameSize(self.lm_input_length*self.lm_input_length*3)
		pre_lm_manip.setWaitForConfigInput(True)
		pre_lm_manip.inputImage.setQueueSize(1)
		pre_lm_manip.inputImage.setBlocking(False)


		# ==== Landmark Detection ====
		print(f"Creating Hand Landmark Neural Network ({'1 thread' if self.lm_nb_threads == 1 else '2 threads'})...")          
		lm_nn = pipeline.create(dai.node.NeuralNetwork)

		# Properties
		lm_nn.setBlobPath(self.landmark_model)
		lm_nn.setNumInferenceThreads(self.lm_nb_threads)

		# ==== Pipeline Linking ====
		print("Pipeline Linking...")

		# Linking Camera RGB
		cam_out = pipeline.createXLinkOut()
		cam_out.setStreamName("RGB preview")
		cam_out.input.setQueueSize(1)
		cam_out.input.setBlocking(False)
		cam.preview.link(cam_out.input)
		
		# Linking Stereo and Depth Camera
		stereo_out = pipeline.createXLinkOut()
		stereo_out.setStreamName("stereo")
		stereo_out.input.setQueueSize(1)
		stereo_out.input.setBlocking(False)
		# stereo.disparity.link(stereo_out.input)
		stereo.depth.link(stereo_out.input)

		left.out.link(stereo.left) #Connect Left Stero Camera to the Stero Node
		right.out.link(stereo.right) #Connect Right Stero Camera to the Stero Node
		stereo.depth.link(spatial_location_calculator.inputDepth) #Connect Left Stero Camera to the Stero Node
		manager_script.outputs['spatial_location_config'].link(spatial_location_calculator.inputConfig) #Set by connection Spatial Location Configuration node
		#NOTE: dato interessante?
		spatial_location_calculator.out.link(manager_script.inputs['spatial_data']) #Connect spatial_location Data to Manager Script Spatial Data section

		# Linking Pre-processing Palm Detection Image Manipulation
		cam.preview.link(pre_pd_manip.inputImage) #Connect RGb Camera Frames to the Pre-processing Palm Detection Image Manipulation
		manager_script.outputs['pre_pd_manip_cfg'].link(pre_pd_manip.inputConfig) #Set by connection Pre-processing Palm Detection Image Manipulation Configuration

		# Linking Palm Detection
		pre_pd_manip.out.link(pd_nn.input) #Connect Pre-processing Palm Detection Image Manipulation to the Palm Detection Node

		# Linking Post-processing Palm Detection
		pd_nn.out.link(post_pd_nn.input) #Connect Palm Detection Node to the Post-processing Palm Detection
		post_pd_nn.out.link(manager_script.inputs['from_post_pd_nn']) #Connect Post-processing Palm Detection to the Manager Script Palm Section

		# Linking Pre-processing Landmark Image Manipulation
		cam.preview.link(pre_lm_manip.inputImage) #Connect RGb Camera Frames to the Pre-processing Landmarks Image Manipulation
		manager_script.outputs['pre_lm_manip_cfg'].link(pre_lm_manip.inputConfig) #Set by connection Pre-processing Landmark Image Manipulation Configuration 

		# Linking Landmark Detection
		pre_lm_manip.out.link(lm_nn.input) #Connect Pre-processing Landmark to Landmark Detection
		lm_nn.out.link(manager_script.inputs['from_lm_nn']) #Connect Landmark Detection to the Manager Script Landmark Section

		# Linking Manager Script 
		manager_out = pipeline.createXLinkOut()
		manager_out.setStreamName("manager_out")
		manager_script.outputs['host'].link(manager_out.input) #Connect Manager Script to the Host

		print("Pipeline created.")
		return pipeline
    # ------------------------------------------------------------------------------------------------------------------  
    
	def build_manager_script(self):
		'''
		The code of the scripting node 'manager_script' depends on :
		    - the score threshold,
		    - the video frame shape
		So we build this code from the content of the file template_manager_script_*.py which is a python template
		'''
		# Read the template from SOLO
		with open(self.manager_script_solo, 'r') as file:
			template = Template(file.read())

		# Perform the substitution
		code = template.substitute(
		            _TRACE1 = "node.warn" if self.trace & 1 else "#",
		            _TRACE2 = "node.warn" if self.trace & 2 else "#",
		            _pd_score_thresh = self.pd_score_thresh,
		            _lm_score_thresh = self.lm_score_thresh,
		            _pad_h = self.pad_h,
		            _img_h = self.img_h,
		            _img_w = self.img_w,
		            _frame_size = self.frame_size,
		            _crop_w = self.crop_w,
		            _IF_XYZ = "" if self.xyz else '"""',
		            _IF_USE_HANDEDNESS_AVERAGE = "" if self.use_handedness_average else '"""',
		            _single_hand_tolerance_thresh = self.single_hand_tolerance_thresh,
		            _IF_USE_SAME_IMAGE = "" if self.use_same_image else '"""',
		            _IF_USE_WORLD_LANDMARKS = "" if self.use_world_landmarks else '"""',
		)
		
		# Remove comments and empty lines
		code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
		code = re.sub(r'#.*', '', code)
		code = re.sub('\n\s*\n', '\n', code)
		
		# For debugging
		if self.trace & 8:
			with open("tmp_code.py", "w") as file:
				file.write(code)

		return code
    # ------------------------------------------------------------------------------------------------------------------

	def getNeuralNetwork(self):
		#Define Script Folder Path and Get Blob files from Trained Neural Network
		SCRIPT_DIR = "/home/ema/Desktop/depthai_marco_scripts"

		PALM_DETECTION_MODEL = str(SCRIPT_DIR + "/models/palm_detection_sh4.blob")
		if not Path(PALM_DETECTION_MODEL).exists():
			raise FileNotFoundError(f'Required palm_detection_sh4.blob file/s not found!')
		else:
			print(f"> Palm Detection blob: {PALM_DETECTION_MODEL}")

		LANDMARK_MODEL_LITE = str(SCRIPT_DIR + "/models/hand_landmark_lite_sh4.blob")
		if not Path(LANDMARK_MODEL_LITE).exists():
			raise FileNotFoundError(f'Required hand_landmark_lite_sh4.blob file/s not found!')
		else:
			print(f"> Landmark blob: {LANDMARK_MODEL_LITE}")

		DETECTION_POSTPROCESSING_MODEL = str(SCRIPT_DIR + "/custom_models/PDPostProcessing_top2_sh1.blob")
		if not Path(DETECTION_POSTPROCESSING_MODEL).exists():
			raise FileNotFoundError(f'Required PDPostProcessing_top2_sh1.blob file/s not found!')
		else:
			print(f"> Post Processing Palm Detection blob: {DETECTION_POSTPROCESSING_MODEL}")

		TEMPLATE_MANAGER_SCRIPT_SOLO = str(SCRIPT_DIR + "/Pipeline/template_manager_script_solo.py")
		if not Path(TEMPLATE_MANAGER_SCRIPT_SOLO).exists():
			raise FileNotFoundError(f'Required template_manager_script_solo.py file/s not found!')
		else:
			print(f"> Manager Script: {TEMPLATE_MANAGER_SCRIPT_SOLO}")

		return PALM_DETECTION_MODEL, LANDMARK_MODEL_LITE, DETECTION_POSTPROCESSING_MODEL, TEMPLATE_MANAGER_SCRIPT_SOLO
    # ------------------------------------------------------------------------------------------------------------------

	def exit(self):
		self.device.close()
		print("Closing Camera...")
	# ------------------------------------------------------------------------------------------------------------------

	def getFrame(self):
		'''If rgb queue empty, this functoin returns None'''
		frame_msg = self.cam_video_preview.get()
		return frame_msg.getCvFrame() if frame_msg is not None else None
	# ------------------------------------------------------------------------------------------------------------------

	def getDepthFrame(self):
		'''If depth queue empty, this function returns None'''
		depth_msg = self.stereo_queue.get() # depth frame returns values in mm (millimeter)
		return depth_msg.getFrame() if depth_msg is not None else None
	# ------------------------------------------------------------------------------------------------------------------

	def getManagerScriptOutput(self):
		temp = self.manager_out.get()
		return temp.getData()
	# ------------------------------------------------------------------------------------------------------------------

	def getDevice(self):
		return self.device
	# ------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	temp = CustomPipeline()
