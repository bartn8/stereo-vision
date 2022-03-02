import cv2
import numpy as np
from pyrSGM import census5x5_SSE, median3x3_SSE, costMeasureCensus5x5_xyd_SSE, matchWTA_SSE, aggregate_SSE
import os


left = cv2.imread(os.path.join("test_data", "view1_crop.png"), cv2.IMREAD_GRAYSCALE)
right = cv2.imread(os.path.join("test_data", "view5_crop.png"), cv2.IMREAD_GRAYSCALE)

h,w = left.shape[:2]
h,w = int(h), int(w)
dispCount = int(128)

dsi = np.zeros((h,w,dispCount), dtype=np.uint16)

leftCensus = np.zeros(left.shape, dtype=np.uint32)
rightCensus = np.zeros(left.shape, dtype=np.uint32)

census5x5_SSE(left, leftCensus, w, h)
census5x5_SSE(right, rightCensus, w, h)

cv2.imwrite(os.path.join("test_data", "census1.png"), leftCensus.astype(np.uint8))
cv2.imwrite(os.path.join("test_data", "census5.png"), rightCensus.astype(np.uint8))

costMeasureCensus5x5_xyd_SSE(leftCensus, rightCensus, dsi, w, h, dispCount, 4)

dsiAgg = np.zeros((h,w,dispCount), dtype=np.uint16)

aggregate_SSE(left, dsi, dsiAgg, w, h, dispCount, 7, 17, 0.25, 50)

dispImg = np.zeros((h,w), dtype=np.float32)

matchWTA_SSE(dsiAgg, dispImg, w,h,dispCount,float(0.95))

dispImgfiltered = np.zeros((h,w), dtype=np.float32)
median3x3_SSE(dispImg, dispImgfiltered, w, h)

cv2.imwrite(os.path.join("test_data", "disp1.png"), dispImgfiltered.astype(np.uint8))
