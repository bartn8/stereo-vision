import cv2
import numpy as np
from pyrSGM import census5x5_SSE, median3x3_SSE, costMeasureCensus5x5_xyd_SSE, matchWTA_SSE, aggregate_SSE, subPixelRefine
import os
import time

def rsgm_double_crop(img):
    _,w = img.shape[:2]
    if w % 16 != 0:
        c = w % 16
        cl = c // 2
        cr = c - cl
        img = img[:, cl:w-cr]
    return img

def gt_resize_scaling(left, right, scale_factor):
    h,w = left.shape[:2]
    left = cv2.resize(left, (w//scale_factor,h//scale_factor))

    h,w = right.shape[:2]
    right = cv2.resize(right, (w//scale_factor,h//scale_factor))


    return left, right

def my_linear_contrast_stretching(gray_image):  
        pmin,pmax = np.min(gray_image), np.max(gray_image)
        image = gray_image.copy().astype(np.float32)
        image -= pmin
        image *= 255.0 / (pmax-pmin)
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)

left = (cv2.imread(os.path.join("test_data", "im0.png"), cv2.IMREAD_GRAYSCALE))
right = (cv2.imread(os.path.join("test_data", "im1.png"), cv2.IMREAD_GRAYSCALE))

left, right = gt_resize_scaling(left, right, 1)

left = rsgm_double_crop(left)
right = rsgm_double_crop(right)

h,w = left.shape[:2]
h,w = int(h), int(w)
dispCount = int(256)

startTime = time.time()

for i in range(5):
    dsi = np.zeros((h,w,dispCount), dtype=np.uint16)

    leftCensus = np.zeros(left.shape, dtype=np.uint32)
    rightCensus = np.zeros(left.shape, dtype=np.uint32)

    census5x5_SSE(left, leftCensus, w, h)
    census5x5_SSE(right, rightCensus, w, h)

    costMeasureCensus5x5_xyd_SSE(leftCensus, rightCensus, dsi, w, h, dispCount, 4)

    dsiAgg = np.zeros((h,w,dispCount), dtype=np.uint16)

    aggregate_SSE(left, dsi, dsiAgg, w, h, dispCount, 11, 17, 0.5, 35)

    dispImg = np.zeros((h,w), dtype=np.float32)

    matchWTA_SSE(dsiAgg, dispImg, w,h,dispCount,float(0.95))
    subPixelRefine(dsiAgg, dispImg, w,h,dispCount,0)

    dispImgfiltered = np.zeros((h,w), dtype=np.float32)
    median3x3_SSE(dispImg, dispImgfiltered, w, h)

print(((time.time()-startTime)*1000)/5)

dispImgfiltered[dispImgfiltered<=0] = 0

dmap_strech = my_linear_contrast_stretching(dispImgfiltered)
dmaprgb = cv2.applyColorMap(dmap_strech, cv2.COLORMAP_VIRIDIS)


cv2.imwrite(os.path.join("test_data", "census1.png"), leftCensus.astype(np.uint8))
cv2.imwrite(os.path.join("test_data", "census5.png"), rightCensus.astype(np.uint8))
cv2.imwrite(os.path.join("test_data", "disp1.png"), dmaprgb)