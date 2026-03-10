#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        # 1. Interpreter load karein (Inference engine)
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        # 2. Memory allocate karein
        self.interpreter.allocate_tensors()

        # 3. Input aur Output ki details nikaalein (Index aur Shape janne ke liye)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        # Input tensor ka index nikaalein
        input_details_index = self.input_details[0]['index']

        # Landmark list ko model ke liye taiyar karein (Float32 aur sahi shape)
        input_data = np.array([landmark_list], dtype=np.float32)

        # Model ke andar data daalein
        self.interpreter.set_tensor(input_details_index, input_data)

        # Model ko "Chalaayein" (Inference)
        self.interpreter.invoke()

        # Result (Output) bahar nikaalein
        output_details_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_index)

        # Sabse zyada probability wali class ka ID return karein (Argmax)
        result_index = np.argmax(np.squeeze(result))

        return result_index