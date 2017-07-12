import numpy as np
import argparse
import cv2
import math
from collections import deque

img = None
orig = None
roi = None
roi2, roi2_init = None,None

kernel = np.array([[0, 0, 1, 1, 0, 0],
                   [0, 1, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 0],
                   [0, 0, 1, 1, 0, 0]],dtype=np.uint8)

ix,iy = 0,0
draw = False
rad_thresh = 15

def getArguements():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    args = vars(ap.parse_args())
    return args

def resize(img,width=400.0):
    r = float(width) / img.shape[0]
    dim = (int(img.shape[1] * r), int(width))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

def selectROI(event, x, y, flag, param):
    global img, ix, iy, draw, orig, roi
    if event == cv2.EVENT_LBUTTONDOWN:
        ix = x
        iy = y
        draw = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if draw:
            img = cv2.rectangle(orig.copy(), (ix, iy), (x, y), (255, 0, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        if draw:
            x1 = max(x, ix)
            y1 = max(y, iy)
            ix = min(x, ix)
            iy = min(y, iy)
            roi = orig[iy:y1, ix:x1]
        draw = False

def getROIvid(frame, winName = 'input'):
    global img, orig, roi
    roi = None
    img = frame.copy()
    orig = frame.copy()
    cv2.namedWindow(winName)
    cv2.setMouseCallback(winName, selectROI)
    while True:
        cv2.imshow(winName, img)
        if roi is not None:
            cv2.destroyWindow(winName)
            return roi

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cv2.destroyWindow(winName)
            break

    return roi

def getROIext(image,winName = 'input'):
    global img, orig, roi2, roi2_init
    img = image.copy()
    orig = image.copy()
    cv2.namedWindow(winName)
    cv2.setMouseCallback(winName, selectROI)
    while True:
        cv2.imshow(winName, img)
        if roi is not None:
            cv2.destroyWindow(winName)
            return roi

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cv2.destroyWindow(winName)
            break

    return roi


def getLimits(roi):
    limits = None
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(roi)
    limits = [(int(np.amax(h)), int(np.amax(s)), 255), (int(np.amin(h)), int(np.amin(s)), int(np.amin(v)))]
    return limits

def applyMorphTransforms(mask):
    global kernel
    lower = 100
    upper = 255

    #mask = cv2.inRange(mask, lower, upper)
    mask = cv2.GaussianBlur(mask, (11, 11), 5)
    mask = cv2.inRange(mask, lower, upper)
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, np.ones((5, 5)))

    return mask

def applyMorphTransforms2(backProj):
    global kernel
    lower = 50
    upper = 255
    mask = cv2.inRange(backProj, lower, upper)
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, np.ones((3, 3)))
    mask = cv2.GaussianBlur(mask, (11, 11), 5)
    mask = cv2.inRange(mask, lower, upper)
    return mask



def detectBallThresh(frame,limits):
    global rad_thresh
    upper = limits[0]
    lower = limits[1]
    center = None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = applyMorphTransforms(mask)
    cv2.imshow('mask', mask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    flag = False
    i=0
    if len(cnts) > 0:
        for i in range(len(cnts)):
            (_, radius) = cv2.minEnclosingCircle(cnts[i])
            if radius < rad_thresh and radius > 5:
                flag = True
                break
        if not flag:
            return None, None

        M = cv2.moments(cnts[i])
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return center, cnts[i]
    else:
        return None, None

def detectBallHB(frame, roi):
    global rad_thresh
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # convert to HSV colour space

    roiHist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backProj = cv2.calcBackProject([hsv], [0, 1], roiHist, [0, 180, 0, 256], 1)
    mask = cv2.inRange(backProj, 50, 255)
    mask = cv2.erode(mask, np.ones((5, 5)))
    mask = cv2.dilate(mask, np.ones((5, 5)))

    cv2.imshow('mask',mask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    i = 0
    if len(cnts) > 0:
        for i in range(len(cnts)):
            (_, radius) = cv2.minEnclosingCircle(cnts[i])
            if radius < rad_thresh and radius>5:
                break
        M = cv2.moments(cnts[i])
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return center, cnts[i]
    else:
        return None, None

def kalmanFilter(meas):
    pred = np.array([],dtype=np.int)
    #mp = np.asarray(meas,np.float32).reshape(-1,2,1)  # measurement
    tp = np.zeros((2, 1), np.float32)  # tracked / prediction

    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.00003
    for mp in meas:
        mp = np.asarray(mp,dtype=np.float32).reshape(2,1)
        kalman.correct(mp)
        tp = kalman.predict()
        np.append(pred,[int(tp[0]),int(tp[1])])

    return pred

def removeBG(frame, fgbg):
    bg_mask = fgbg.apply(frame)
    bg_mask = cv2.dilate(bg_mask, np.ones((5, 5)))
    frame = cv2.bitwise_and(frame, frame, mask=bg_mask)
    return frame

def getHist(frame):
    roi_hist_A, roi_hist_B = None, None

    if roi_hist_A is None:
        roi = getROIvid(frame,'input team A')
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        roi_hist_A = cv2.calcHist([roi],[0,1],None,[180,256],[0,180,0,256])
        roi_hist_A = cv2.normalize(roi_hist_A, roi_hist_A, 0, 255, cv2.NORM_MINMAX)

    if roi_hist_B is None:
        roi = getROIvid(frame, 'input team B')
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist_B = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        roi_hist_B = cv2.normalize(roi_hist_B, roi_hist_B, 0, 255, cv2.NORM_MINMAX)

    return roi_hist_A, roi_hist_B