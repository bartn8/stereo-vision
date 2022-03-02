import cv2
import numpy as np
from pyrSGM import census5x5_SSE, median3x3_SSE, costMeasureCensus5x5_xyd_SSE, matchWTA_SSE
import os


left = cv2.imread(os.path.join("test_data", "view1_crop.png"), cv2.IMREAD_GRAYSCALE)
right = cv2.imread(os.path.join("test_data", "view5_crop.png"), cv2.IMREAD_GRAYSCALE)

