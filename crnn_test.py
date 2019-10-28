# -*- coding: utf-8 -*-
import cv2

from crnn.onnx import Crnn


crnn = Crnn('./crnn/crnn.onnx', cal_score = True)

image = cv2.imread('./images/test.jpg')
image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

results = crnn.recognize(image)
print(results)