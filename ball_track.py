import numpy as np
import cv2
import math
from collections import deque
import track_utils

frame = None
limits = None
center = None
prevgrad = 0
passes = 0
pts = deque()
orig = None

def trackBall():
    global grad, prevgrad, passes, center, orig
    pts.appendleft(center)

    for i in xrange(1, len(pts)):
        cv2.line(orig, pts[i-1], pts[i], (0, 0, 255), 2, cv2.LINE_AA)

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

    args = track_utils.getArguements()

    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])

    fgbg = cv2.createBackgroundSubtractorMOG2(history=20,detectShadows=False)
    while True:
        (grabbed, frame) = camera.read()

        if args.get("video") and not grabbed:
            break

        frame = track_utils.resize(frame)
        orig = frame.copy()

        frame = track_utils.removeBG(frame,fgbg)

        if limits is not None:
            center,cnt = track_utils.detectBallThresh(frame,limits)
            if cnt is not None:
                (x,y),radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(orig,(int(x),int(y)),int(radius),(255,255,0),2)
                cv2.circle(orig,center,2,(0,0,255),-1)
                trackBall()

        cv2.imshow('frame', orig)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord('i') and limits is None:
            roi = track_utils.getROIvid(orig, 'input ball')

            if roi is not None:
                limits = track_utils.getLimits(roi)

    camera.release()
    cv2.destroyAllWindows()
