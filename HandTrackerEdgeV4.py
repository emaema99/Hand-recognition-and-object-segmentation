#!/usr/bin/env python3

import numpy as np
import mediapipe_utils as mpu
import marshal

class HandTracker:
    '''
    Class for initialize and run the HandTracker algorithm from Mediapipe.
    '''
    def __init__(self, frame_size, pad_h, pad_w, lm_score_thresh):
        self.frame_size = frame_size
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.lm_score_thresh = lm_score_thresh
        print("HandTracker is running ...")
    # ---------------------------------------------------------------------------------------------------

    def extract_hand_data(self, res, hand_idx):
        '''
        Returns data info such as depth, landmarks and gestures.
        '''
        hand = mpu.HandRegion()

        hand.rect_x_center_a = res["rect_center_x"][hand_idx] * self.frame_size
        hand.rect_y_center_a = res["rect_center_y"][hand_idx] * self.frame_size
        hand.rect_w_a = hand.rect_h_a = res["rect_size"][hand_idx] * self.frame_size

        hand.rotation = res["rotation"][hand_idx] 
        hand.rect_points = mpu.rotated_rect_to_points(hand.rect_x_center_a, hand.rect_y_center_a, hand.rect_w_a, hand.rect_h_a, hand.rotation)

        hand.lm_score = res["lm_score"][hand_idx]
        hand.handedness = res["handedness"][hand_idx]
        hand.label = "right" if hand.handedness > 0.5 else "left"

        hand.norm_landmarks = np.array(res['rrn_lms'][hand_idx]).reshape(-1,3)
        hand.landmarks = (np.array(res["sqn_lms"][hand_idx]) * self.frame_size).reshape(-1,2).astype(np.int32)

        hand.xyz = np.array(res["xyz"][hand_idx])
        hand.xyz_zone = res["xyz_zone"][hand_idx]
        hand.xyz_lms = np.array(res["xyz_lms"][hand_idx])

        # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
        if self.pad_h > 0:
            hand.landmarks[:,1] -= self.pad_h
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][1] -= self.pad_h

        if self.pad_w > 0:
            hand.landmarks[:,0] -= self.pad_w
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][0] -= self.pad_w

        # World landmarks (Passed TRUE)
        hand.world_landmarks = np.array(res["world_lms"][hand_idx]).reshape(-1, 3)

        hand = mpu.recognize_gesture(hand)

        return hand
    # ---------------------------------------------------------------------------------------------------

    def getHandsData(self, handsData):
        '''
        Get Results from Manager Script "template_manager_script_solo.py"
        '''
        handsResult = marshal.loads(handsData)
        hands = []

        for i in range(len(handsResult.get("lm_score",[]))):
            hand = self.extract_hand_data(handsResult, i)
            hands.append(hand)

        return hands
    # ---------------------------------------------------------------------------------------------------
