import numpy as np
import argparse
import cv2
import math
from collections import deque

frame = None
orig = None
roi = None
ix = 0
iy = 0
limits = None
draw = False
center = None
prevgrad = 0
passes = 0
pts = deque()


def getArguements():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    args = vars(ap.parse_args())
    return args


def selectROI(event, x, y, flag, param):
    global roi, frame, orig, ix, iy, draw
    copy = orig.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        ix = x
        iy = y
        draw = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if draw:
            frame = cv2.rectangle(copy, (ix, iy), (x, y), (255, 0, 0), 2)
            copy = orig.copy()

    elif event == cv2.EVENT_LBUTTONUP:

        if draw:
            x1 = max(x, ix)
            y1 = max(y, iy)
            ix = min(x, ix)
            iy = min(y, iy)
            roi = orig[iy:y1, ix:x1]
        draw = False


def resize():
    global frame
    r = 400.0 / frame.shape[0]
    dim = (int(frame.shape[1] * r), 400)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def getLimits():
    global limits, roi, frame

    cv2.namedWindow('input')
    cv2.setMouseCallback('input', selectROI)
    while True:
        cv2.imshow('input', frame)
        if not (roi is None):
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(roi)
            limits = [(int(np.amax(h)), int(np.amax(s)), 255), (int(np.amin(h)), int(np.amin(s)), int(np.amin(v)))]

            print limits
            cv2.destroyWindow('input')
            break

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cv2.destroyWindow('input')
            break

def detectBall():
    global center,limits,frame
    upper = limits[0]
    lower = limits[1]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow('mask', mask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        trackBall()
        if radius > 5:
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)


def trackBall():
    global grad, prevgrad, passes
    pts.appendleft(center)

    for i in xrange(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), 2, cv2.LINE_AA)

    l = len(pts)
    if l >= 10:
        grad = np.arctan2((pts[9][1] - pts[0][1]), (pts[9][0] - pts[0][0]))
        grad = grad * 180 / np.pi
        grad %= 360

        if(math.fabs(grad - prevgrad) >= 20) and math.sqrt((pts[9][1] - pts[0][1]) ** 2 + (pts[9][0] - pts[0][0]) ** 2) >= 20:
            print('Ball Passed ' + str(passes))
            passes += 1
        prevgrad = grad


if __name__ == '__main__':
    args = getArguements()

    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])

    while True:
        (grabbed, frame) = camera.read()

        if args.get("video") and not grabbed:
            break

        resize()

        frame = cv2.bilateralFilter(frame, 5, 15, 15)

        orig = frame.copy()
        if limits is not None:
            detectBall()

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord('i') and limits is None:
            getLimits()

    camera.release()
    cv2.destroyAllWindows()
