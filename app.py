#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import os
from playsound import playsound
import threading

import cv2 as cv
import numpy as np
import mediapipe as mp

import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

#TYe function argparse library use karke terminal (command line) se user dwara diye gaye inputs ko read karta hai.
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():

    is_playing = False
    audio_path = "/home/dikki/Desktop/hand-gesture/FAHHH.m4a"

    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device # Camera device index
    cap_width = args.width # Camera capture width
    cap_height = args.height # Camera capture height


    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    # mp_hands = mp.solutions.hands
    # hands = mp_hands.Hands(
    #     static_image_mode=use_static_image_mode,
    #     max_num_hands=1,
    #     min_detection_confidence=min_detection_confidence,
    #     min_tracking_confidence=min_tracking_confidence,
    # )
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO, # Webcam ke liye VIDEO mode sahi hai
        num_hands=2,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        keypoint_classifier = KeyPointClassifier()  
        # KeyPointClassifier - Ye hath ki "Static" (ruki hui) position ko pehchanta hai. 
        # Jaise agar aapki ungliyan band hain, to ye batayega ki ye "Fist" (mutthi) hai.

        point_history_classifier = PointHistoryClassifier()

        # PointHistoryClassifier - Ye hath ki "Dynamic" (chalti hui) position ko pehchanta hai. 
        # Jaise agar aap apni ungli ko ek taraf se dusri taraf le jaate hain, to ye batayega ki ye "Pointing" (nishaan dena) hai.   

        # Read labels - Naam deta hai ki kaunsa gesture hai, jaise "Fist", "Open Hand", "Pointing" etc.
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
        with open(
                'model/point_history_classifier/point_history_classifier_label.csv',
                encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            point_history_classifier_labels = [
                row[0] for row in point_history_classifier_labels
            ]

        # FPS Measurement ########################################################
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        # Coordinate history #################################################################
        history_length = 16
        point_history = deque(maxlen=history_length)
        # Used to store the history of the coordinates of the index finger tip.

        # Finger gesture history ################################################
        finger_gesture_history = deque(maxlen=history_length)
        # Used to store the history of the finger gestures.

        #  ########################################################################
        mode = 0

        while True:
            fps = cvFpsCalc.get()

            found_gesture_4 = False

            # Process Key (ESC: end) #################################################
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = select_mode(key, mode)

            # Camera capture #####################################################
            ret, image = cap.read() 
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # Detection implementation #############################################################
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            timestamp_ms = int(time.time() * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            #  ####################################################################
            if results.hand_landmarks:
                for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                    # Yahan hand_landmarks ek list hai, object nahi
                    # Isliye humein helper functions ko thoda modify karna hoga



                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)
                    
                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, point_history)
                    # Write to the dataset file - add gesture data to CSV file for training
                    logging_csv(number, mode, pre_processed_landmark_list,
                                pre_processed_point_history_list)

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 2:  # Point gesture
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    if hand_sign_id == 4:
                        found_gesture_4 = True

                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(
                            pre_processed_point_history_list)

                    # Calculates the gesture IDs in the latest detection
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(
                        finger_gesture_history).most_common()
                    # "Majority Vote" ki tarah kaam karta hai.
                    # Agar pichle 16 mein se zyadaatar frames mein "Clockwise" gesture dikha hai, 
                    # to bhale hi 1-2 frames mein galti hui ho, screen par "Clockwise" hi dikhaya jayega.

                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[most_common_fg_id[0][0]],
                    )
            else:
                point_history.append([0, 0])

            if found_gesture_4:
                if not is_playing:
                    os.system(f"cvlc --loop '{audio_path}' > /dev/null 2>&1 &")
                    is_playing = True
                    print("Continuous sound started...")
            else:
                if is_playing:
                    # Agar haath hat gaya (no landmarks) ya ID 4 nahi hai
                    os.system("pkill -f vlc")
                    is_playing = False
                    print("Sound stopped.")

            debug_image = draw_point_history(debug_image, point_history)
            debug_image = draw_info(debug_image, fps, mode, number)

            # Screen reflection #############################################################
            cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    # for _, landmark in enumerate(landmarks.landmark):
    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    # for _, landmark in enumerate(landmarks.landmark):
    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list)) 
    # map(normalize_, temp_landmark_list) - har element par ye function lagata hai.

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))
    # Machine model data ko ek "Flat Vector" yaani ek single array
    # mtlb lambi array mein padta hai.
    #. Use ye samajh nahi aata ki kaunsa x aur y kis group ka hai; 
    # use bas ek ke baad ek saari values chahiye hoti hain.

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv' #Location of CSV file.
        with open(csv_path, 'a', newline="") as f: #Open file in append mode.
            writer = csv.writer(f) # Ye csv library ka ek tool (writer object) taiyar karta hai
            # jo data ko sahi CSV format mein likhne mein madad karta hai.
            writer.writerow([number, *landmark_list]) #Ye asli data ko file mein ek row (line) ke roop mein likhta hai.
    if mode == 2 and (0 <= number <= 9): 
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f: #Open file in append mode.
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # 1. Saare connections define karein (Joints ke beech ki lines)
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),          # Index
            (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
            (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),  # Little
            (0, 17)                                  # Palm
        ]

        # 2. Loop se saari lines draw karein
        for connection in HAND_CONNECTIONS:
            pt1 = tuple(landmark_point[connection[0]])
            pt2 = tuple(landmark_point[connection[1]])
            # Shadow effect ke liye pehle moti black line, phir patli white
            cv.line(image, pt1, pt2, (0, 0, 0), 6)
            cv.line(image, pt1, pt2, (255, 255, 255), 2)

        # 3. Loop se saare joints (circles) draw karein
        for index, landmark in enumerate(landmark_point):
            # Ungliyon ke siron (tips) ke liye bada circle
            size = 8 if index in [4, 8, 12, 16, 20] else 5
            cv.circle(image, (landmark[0], landmark[1]), size, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), size, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1) 

    info_text = handedness[0].category_name
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
