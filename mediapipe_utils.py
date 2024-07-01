#!/usr/bin/env python3

from cv2 import __version__, dnn, getAffineTransform, warpAffine
from numpy import array, dot, seterr, argmax, tile, linalg, arccos, float32, exp, degrees
from re import sub
from collections import namedtuple
from math import ceil, sqrt, floor, sin, cos, atan2, gcd, pi

seterr(over = 'ignore')

class HandRegion:
    """
        >> Attributes:

        pd_score:                           detection score
        pd_box:                             detection box [x, y, w, h], normalized [0,1] in the squared image
        pd_kps:                             detection keypoints coordinates [x, y], normalized [0,1] in the squared image
        rect_x_center, rect_y_center:       center coordinates of the rotated bounding rectangle, normalized [0,1] in the squared image
        rect_w, rect_h:                     width and height of the rotated bounding rectangle, normalized in the squared image (may be > 1)
        rotation:                           rotation angle of rotated bounding rectangle with y-axis in radian
        rect_x_center_a, rect_y_center_a:   center coordinates of the rotated bounding rectangle, in pixels in the squared image
        rect_w, rect_h:                     width and height of the rotated bounding rectangle, in pixels in the squared image
        rect_points:                        list of the 4 points coordinates of the rotated bounding rectangle, in pixels 
                                            expressed in the squared image during processing,
                                            expressed in the source rectangular image when returned to the user
        lm_score:                           global landmark score
        norm_landmarks:                     3D landmarks coordinates in the rotated bounding rectangle, normalized [0,1]
        landmarks:                          2D landmark coordinates in pixel in the source rectangular image
        world_landmarks:                    3D landmark coordinates in meter
        handedness:                         float between 0. and 1., > 0.5 for right hand, < 0.5 for left hand,
        label:                              "left" or "right", handedness translated in a string,
        xyz:                                real 3D world coordinates of the wrist landmark, or of the palm center (if landmarks are not used),
        xyz_zone:                           (left, top, right, bottom), pixel coordinates in the source rectangular image 
                                            of the rectangular zone used to estimate the depth
        gesture:                            Tells if grasping action is recognised or not.
        """
    def __init__(self, pd_score=None, pd_box=None, pd_kps=None):
        self.pd_score = pd_score # Palm detection score 
        self.pd_box = pd_box # Palm detection box [x, y, w, h] normalized
        self.pd_kps = pd_kps # Palm detection keypoints
        self.gesture = "-"

    def get_rotated_world_landmarks(self):
        world_landmarks_rotated = self.world_landmarks.copy()
        sin_rot = sin(self.rotation)
        cos_rot = cos(self.rotation)
        rot_m = array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
        world_landmarks_rotated[:,:2] = dot(world_landmarks_rotated[:,:2], rot_m)
        return world_landmarks_rotated

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))
# -------------------------------------------------------------------------------------------------------------

class HandednessAverage:
    """
    Class used to store the average handeness (left/right).
    Handedness inferred by the landmark model is not perfect. For certain poses, it is not rare that the model thinks 
    that a right hand is a left hand (or vice versa). Instead of using the last inferred handedness, we prefer to use the average 
    of the inferred handedness on the last frames. This gives more robustness.
    """
    def __init__(self):
        self._total_handedness = 0
        self._nb = 0

    def update(self, new_handedness):
        self._total_handedness += new_handedness
        self._nb += 1
        return self._total_handedness / self._nb
    
    def reset(self):
        self._total_handedness = self._nb = 0
# -------------------------------------------------------------------------------------------------------------

SSDAnchorOptions = namedtuple(
    'SSDAnchorOptions',[
        'num_layers',
        'min_scale',
        'max_scale',
        'input_size_height',
        'input_size_width',
        'anchor_offset_x',
        'anchor_offset_y',
        'strides',
        'aspect_ratios',
        'reduce_boxes_in_lowest_layer',
        'interpolated_scale_aspect_ratio',
        'fixed_anchor_size'
        ])

def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (min_scale + max_scale) / 2
    else:
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1)
# -------------------------------------------------------------------------------------------------------------

def generate_anchors(options):
    """
    Options : SSDAnchorOptions
    # https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
    """
    anchors = []
    layer_id = 0
    n_strides = len(options.strides)

    while layer_id < n_strides:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []
        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id

        while last_same_stride_layer < n_strides and \
                options.strides[last_same_stride_layer] == options.strides[layer_id]:
            scale = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer, n_strides)
            if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios += [1.0, 2.0, 0.5]
                scales += [0.1, scale, scale]
            else:
                aspect_ratios += options.aspect_ratios
                scales += [scale] * len(options.aspect_ratios)
                if options.interpolated_scale_aspect_ratio > 0:
                    if last_same_stride_layer == n_strides -1:
                        scale_next = 1.0
                    else:
                        scale_next = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer+1, n_strides)
                    scales.append(sqrt(scale * scale_next))
                    aspect_ratios.append(options.interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1

        for i,r in enumerate(aspect_ratios):
            ratio_sqrts = sqrt(r)
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = options.strides[layer_id]
        feature_map_height = ceil(options.input_size_height / stride)
        feature_map_width = ceil(options.input_size_width / stride)

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options.anchor_offset_x) / feature_map_width
                    y_center = (y + options.anchor_offset_y) / feature_map_height
                    # new_anchor = Anchor(x_center=x_center, y_center=y_center)
                    if options.fixed_anchor_size:
                        new_anchor = [x_center, y_center, 1.0, 1.0]
                        # new_anchor.w = 1.0
                        # new_anchor.h = 1.0
                    else:
                        new_anchor = [x_center, y_center, anchor_width[anchor_id], anchor_height[anchor_id]]
                        # new_anchor.w = anchor_width[anchor_id]
                        # new_anchor.h = anchor_height[anchor_id]
                    anchors.append(new_anchor)
        
        layer_id = last_same_stride_layer

    return array(anchors)
# -------------------------------------------------------------------------------------------------------------

def generate_handtracker_anchors(input_size_width, input_size_height):
    '''
    https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
    '''
    anchor_options = SSDAnchorOptions(
        num_layers = 4,
        min_scale = 0.1484375,
        max_scale = 0.75,
        input_size_height = input_size_height,
        input_size_width = input_size_width,
        anchor_offset_x = 0.5,
        anchor_offset_y = 0.5,
        strides = [8, 16, 16, 16],
        aspect_ratios = [1.0],
        reduce_boxes_in_lowest_layer = False,
        interpolated_scale_aspect_ratio = 1.0,
        fixed_anchor_size = True)
    
    return generate_anchors(anchor_options)
# -------------------------------------------------------------------------------------------------------------

def decode_bboxes(score_thresh, scores, bboxes, anchors, scale = 128, best_only = False):
    """
    wi, hi : NN input shape
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    # Decodes the detection tensors generated by the model, based on
    # the SSD anchors and the specification in the options, into a vector of
    # detections. Each detection describes a detected object.

    https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt :
    node {
        calculator: "TensorsToDetectionsCalculator"
        input_stream: "TENSORS:detection_tensors"
        input_side_packet: "ANCHORS:anchors"
        output_stream: "DETECTIONS:unfiltered_detections"
        options: {
            [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
            num_classes: 1
            num_boxes: 896
            num_coords: 18
            box_coord_offset: 0
            keypoint_coord_offset: 4
            num_keypoints: 7
            num_values_per_keypoint: 2
            sigmoid_score: true
            score_clipping_thresh: 100.0
            reverse_output_order: true

            x_scale: 128.0
            y_scale: 128.0
            h_scale: 128.0
            w_scale: 128.0
            min_score_thresh: 0.5
            }
        }
    }
    node {
        calculator: "TensorsToDetectionsCalculator"
        input_stream: "TENSORS:detection_tensors"
        input_side_packet: "ANCHORS:anchors"
        output_stream: "DETECTIONS:unfiltered_detections"
        options: {
            [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
            num_classes: 1
            num_boxes: 2016
            num_coords: 18
            box_coord_offset: 0
            keypoint_coord_offset: 4
            num_keypoints: 7
            num_values_per_keypoint: 2
            sigmoid_score: true
            score_clipping_thresh: 100.0
            reverse_output_order: true

            x_scale: 192.0
            y_scale: 192.0
            w_scale: 192.0
            h_scale: 192.0
            min_score_thresh: 0.5
            }
        }
    }

    scores: shape = [number of anchors 896 or 2016]
    bboxes: shape = [ number of anchors x 18], 18 = 4 (bounding box : (cx,cy,w,h) + 14 (7 palm keypoints)
    """
    regions = []
    scores = 1 / (1 + exp(-scores))

    if best_only:
        best_id = argmax(scores)
        if scores[best_id] < score_thresh: return regions
        det_scores = scores[best_id:best_id+1]
        det_bboxes2 = bboxes[best_id:best_id+1]
        det_anchors = anchors[best_id:best_id+1]
    else:
        detection_mask = scores > score_thresh
        det_scores = scores[detection_mask]
        if det_scores.size == 0: return regions
        det_bboxes2 = bboxes[detection_mask]
        det_anchors = anchors[detection_mask]

    det_bboxes = det_bboxes2* tile(det_anchors[:,2:4], 9) / scale + tile(det_anchors[:,0:2],9)
    det_bboxes[:,2:4] = det_bboxes[:,2:4] - det_anchors[:,0:2]
    det_bboxes[:,0:2] = det_bboxes[:,0:2] - det_bboxes[:,3:4] * 0.5

    for i in range(det_bboxes.shape[0]):
        score = det_scores[i]
        box = det_bboxes[i,0:4]
        # Decoded detection boxes could have negative values for width/height due
        # to model prediction. Filter out those boxes
        if box[2] < 0 or box[3] < 0: continue
        kps = []
        # 0 : wrist
        # 1 : index finger joint
        # 2 : middle finger joint
        # 3 : ring finger joint
        # 4 : little finger joint
        # 5 : 
        # 6 : thumb joint
        for kp in range(7):
            kps.append(det_bboxes[i,4+kp*2:6+kp*2])

        regions.append(HandRegion(float(score), box, kps))

    return regions
# -------------------------------------------------------------------------------------------------------------

# Starting from opencv 4.5.4, dnn.NMSBoxes output format changed
cv2_version = __version__.split('.')
v0 = int(cv2_version[0])
v1 = int(cv2_version[1])
v2 = int(sub(r'\D+', '', cv2_version[2]))

if  v0 > 4 or (v0 == 4 and (v1 > 5 or (v1 == 5 and v2 >= 4))):
    def non_max_suppression(regions, nms_thresh):
        # dnn.NMSBoxes(boxes, scores, 0, nms_thresh) needs:
        # boxes = [ [x, y, w, h], ...] with x, y, w, h of type int
        # Currently, x, y, w, h are float between 0 and 1, so we arbitrarily multiply by 1000 and cast to int
        boxes = [ [int(x*1000) for x in r.pd_box] for r in regions]        
        scores = [r.pd_score for r in regions]
        indices = dnn.NMSBoxes(boxes, scores, 0, nms_thresh) # Not using top_k=2 here because it does not give expected result. Bug ?
        return [regions[i] for i in indices]
else:
    def non_max_suppression(regions, nms_thresh):
        boxes = [ [int(x*1000) for x in r.pd_box] for r in regions]        
        scores = [r.pd_score for r in regions]
        indices = dnn.NMSBoxes(boxes, scores, 0, nms_thresh)
        return [regions[i[0]] for i in indices]
# -------------------------------------------------------------------------------------------------------------

def normalize_radians(angle):
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))
# -------------------------------------------------------------------------------------------------------------

def rot_vec(vec, rotation):
    vx, vy = vec
    return [vx * cos(rotation) - vy * sin(rotation), vx * sin(rotation) + vy * cos(rotation)]
# -------------------------------------------------------------------------------------------------------------

def detections_to_rect(regions):
    '''
    https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt
    Converts results of palm detection into a rectangle (normalized by image size)
    that encloses the palm and is rotated such that the line connecting center of
    the wrist and MCP of the middle finger is aligned with the Y-axis of the rectangle:

    node {
      calculator: "DetectionsToRectsCalculator"
      input_stream: "DETECTION:detection"
      input_stream: "IMAGE_SIZE:image_size"
      output_stream: "NORM_RECT:raw_roi"
      options: {
        [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
          rotation_vector_start_keypoint_index: 0  # Center of wrist.
          rotation_vector_end_keypoint_index: 2  # MCP of middle finger.
          rotation_vector_target_angle_degrees: 90
        }
      }
    '''
    target_angle = pi * 0.5 # 90 = pi/2

    for region in regions:
        region.rect_w = region.pd_box[2]
        region.rect_h = region.pd_box[3]
        region.rect_x_center = region.pd_box[0] + region.rect_w / 2
        region.rect_y_center = region.pd_box[1] + region.rect_h / 2

        x0, y0 = region.pd_kps[0] # wrist center
        x1, y1 = region.pd_kps[2] # middle finger
        rotation = target_angle - atan2(-(y1 - y0), x1 - x0)
        region.rotation = normalize_radians(rotation)
# -------------------------------------------------------------------------------------------------------------

def rotated_rect_to_points(cx, cy, w, h, rotation):
    b = cos(rotation) * 0.5
    a = sin(rotation) * 0.5
    p0x = cx - a*h - b*w
    p0y = cy + b*h - a*w
    p1x = cx + a*h - b*w
    p1y = cy - b*h - a*w
    p2x = int(2*cx - p0x)
    p2y = int(2*cy - p0y)
    p3x = int(2*cx - p1x)
    p3y = int(2*cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)

    return [[p0x,p0y], [p1x,p1y], [p2x,p2y], [p3x,p3y]]
# -------------------------------------------------------------------------------------------------------------

def rect_transformation(regions, w, h):
    '''
    w, h : image input shape.
    https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt
    Expands and shifts the rectangle that contains the palm so that it's likely to cover the entire hand.
    node {
    calculator: "RectTransformationCalculator"
    input_stream: "NORM_RECT:raw_roi"
    input_stream: "IMAGE_SIZE:image_size"
    output_stream: "roi"
    options: {
        [mediapipe.RectTransformationCalculatorOptions.ext] {
        scale_x: 2.6
        scale_y: 2.6
        shift_y: -0.5
        square_long: true
        }
    }
    IMHO 2.9 is better than 2.6. With 2.6, it may happen that finger tips stay outside of the bouding rotated rectangle
    '''
    scale_x = 2.9
    scale_y = 2.9
    shift_x = 0
    shift_y = -0.5

    for region in regions:
        width = region.rect_w
        height = region.rect_h
        rotation = region.rotation
        if rotation == 0:
            region.rect_x_center_a = (region.rect_x_center + width * shift_x) * w
            region.rect_y_center_a = (region.rect_y_center + height * shift_y) * h
        else:
            x_shift = (w * width * shift_x * cos(rotation) - h * height * shift_y * sin(rotation)) #/ w
            y_shift = (w * width * shift_x * sin(rotation) + h * height * shift_y * cos(rotation)) #/ h
            region.rect_x_center_a = region.rect_x_center*w + x_shift
            region.rect_y_center_a = region.rect_y_center*h + y_shift

        long_side = max(width * w, height * h)
        region.rect_w_a = long_side * scale_x
        region.rect_h_a = long_side * scale_y
        region.rect_points = rotated_rect_to_points(region.rect_x_center_a, region.rect_y_center_a, region.rect_w_a, region.rect_h_a, region.rotation)
# -------------------------------------------------------------------------------------------------------------

def hand_landmarks_to_rect(hand):
    '''
    Calculates the ROI for the next frame from the current hand landmarks
    '''
    id_wrist = 0
    id_index_mcp = 5
    id_middle_mcp = 9
    id_ring_mcp =13
    
    lms_xy =  hand.landmarks[:,:2]
    # Compute rotation
    x0, y0 = lms_xy[id_wrist]
    x1, y1 = 0.25 * (lms_xy[id_index_mcp] + lms_xy[id_ring_mcp]) + 0.5 * lms_xy[id_middle_mcp]
    rotation = 0.5 * pi - atan2(y0 - y1, x1 - x0)
    rotation = normalize_radians(rotation)
    # Now we work only on a subset of the landmarks
    ids_for_bounding_box = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
    lms_xy = lms_xy[ids_for_bounding_box]
    # Find center of the boundaries of landmarks
    axis_aligned_center = 0.5 * (min(lms_xy, axis=0) + max(lms_xy, axis=0))
    # Find boundaries of rotated landmarks
    original = lms_xy - axis_aligned_center
    c, s = cos(rotation), sin(rotation)
    rot_mat = array(((c, -s), (s, c)))
    projected = original.dot(rot_mat)
    min_proj = min(projected, axis=0)
    max_proj = max(projected, axis=0)
    projected_center = 0.5 * (min_proj + max_proj)
    center = rot_mat.dot(projected_center) + axis_aligned_center
    width, height = max_proj - min_proj
    next_hand = HandRegion()
    next_hand.rect_w_a = next_hand.rect_h_a = 2 * max(width, height)
    next_hand.rect_x_center_a = center[0] + 0.1 * height * s
    next_hand.rect_y_center_a = center[1] - 0.1 * height * c
    next_hand.rotation = rotation
    next_hand.rect_points = rotated_rect_to_points(next_hand.rect_x_center_a, next_hand.rect_y_center_a, next_hand.rect_w_a, next_hand.rect_h_a, next_hand.rotation)

    return next_hand
# -------------------------------------------------------------------------------------------------------------

def warp_rect_img(rect_points, img, w, h):
        src = array(rect_points[1:], dtype=float32) # rect_points[0] is left bottom point !
        dst = array([(0, 0), (h, 0), (h, w)], dtype=float32)
        mat = getAffineTransform(src, dst)

        return warpAffine(img, mat, (w, h))
# -------------------------------------------------------------------------------------------------------------

def distance(a, b):
    """
    a, b: 2 points (in 2D or 3D)
    """
    return linalg.norm(a-b)
# -------------------------------------------------------------------------------------------------------------

def angle(a, b, c):
    '''
    https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates.
    a, b and c : points as array([x, y, z])
    ''' 
    ba = a - b
    bc = c - b
    cosine_angle = dot(ba, bc) / (linalg.norm(ba) * linalg.norm(bc))
    angle = arccos(cosine_angle)

    return degrees(angle)
# -------------------------------------------------------------------------------------------------------------

def find_isp_scale_params(size, resolution, is_height=True):
    """
    Find closest valid size close to 'size' and and the corresponding parameters to setIspScale()
    This function is useful to work around a bug in depthai where ImageManip is scrambling images that have an invalid size
    resolution: sensor resolution (width, height)
    is_height : boolean that indicates if the value 'size' represents the height or the width of the image
    Returns: valid size, (numerator, denominator)
    """
    # We want size >= 288 (first compatible size > lm_input_size)
    if size < 288:
        size = 288

    width, height = resolution

    # We are looking for the list on integers that are divisible by 16 and
    # that can be written like n/d where n <= 16 and d <= 63
    if is_height:
        reference = height 
        other = width
    else:
        reference = width 
        other = height
    size_candidates = {}
    for s in range(288,reference,16):
        f = gcd(reference, s)
        n = s//f
        d = reference//f
        if n <= 16 and d <= 63 and int(round(other * n / d) % 2 == 0):
            size_candidates[s] = (n, d)

    # What is the candidate size closer to 'size' ?
    min_dist = -1
    for s in size_candidates:
        dist = abs(size - s)
        if min_dist == -1:
            min_dist = dist
            candidate = s
        else:
            if dist > min_dist: break
            candidate = s
            min_dist = dist

    return candidate, size_candidates[candidate]
# -------------------------------------------------------------------------------------------------------------

def recognize_gesture(hand):           
    '''
    Finger states: (-1) = unknown, (0) = close, (1) = open.
    Grasping/not grasping state based on fingers state: if >= 3 fingers are open, we assume no grasping happens.
    '''
    d_3_5 = distance(hand.norm_landmarks[3], hand.norm_landmarks[5])
    d_2_3 = distance(hand.norm_landmarks[2], hand.norm_landmarks[3])

    angle0 = angle(hand.norm_landmarks[0], hand.norm_landmarks[1], hand.norm_landmarks[2])
    angle1 = angle(hand.norm_landmarks[1], hand.norm_landmarks[2], hand.norm_landmarks[3])
    angle2 = angle(hand.norm_landmarks[2], hand.norm_landmarks[3], hand.norm_landmarks[4])
    hand.thumb_angle = angle0 + angle1 + angle2

    if hand.thumb_angle > 460 and d_3_5 / d_2_3 > 1.2: 
        hand.thumb_state = 1
    else:
        hand.thumb_state = 0

    if hand.norm_landmarks[8][1] < hand.norm_landmarks[7][1] < hand.norm_landmarks[6][1]:
        hand.index_state = 1
    elif hand.norm_landmarks[6][1] < hand.norm_landmarks[8][1]:
        hand.index_state = 0
    else:
        hand.index_state = -1

    if hand.norm_landmarks[12][1] < hand.norm_landmarks[11][1] < hand.norm_landmarks[10][1]:
        hand.middle_state = 1
    elif hand.norm_landmarks[10][1] < hand.norm_landmarks[12][1]:
        hand.middle_state = 0
    else:
        hand.middle_state = -1

    if hand.norm_landmarks[16][1] < hand.norm_landmarks[15][1] < hand.norm_landmarks[14][1]:
        hand.ring_state = 1
    elif hand.norm_landmarks[14][1] < hand.norm_landmarks[16][1]:
        hand.ring_state = 0
    else:
        hand.ring_state = -1

    if hand.norm_landmarks[20][1] < hand.norm_landmarks[19][1] < hand.norm_landmarks[18][1]:
        hand.little_state = 1
    elif hand.norm_landmarks[18][1] < hand.norm_landmarks[20][1]:
        hand.little_state = 0
    else:
        hand.little_state = -1
    
    # Grasping/not grasping state based on fingers state: if >= 3 fingers are open, no grasping happens (assumption)
    if (hand.thumb_state + hand.index_state + hand.middle_state + hand.ring_state + hand.little_state) < 3:
        hand.is_grasping = True
    else:
        hand.is_grasping = False
 
    return hand
# -------------------------------------------------------------------------------------------------------------
