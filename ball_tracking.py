import numpy as np
import argparse
import cv2

inputMode = False
frame = None
orig = None
roi = None
ix = 0
iy = 0
limits = None
draw = False

def getArguements():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64,
                    help="max buffer size")
    args = vars(ap.parse_args())
    return args

def selectROI(event,x,y,flag,param):
    global roi,frame,inputMode,orig,ix,iy,draw
    copy = orig.copy()
    if event == cv2.EVENT_LBUTTONDOWN and inputMode:
        ix=x
        iy=y
        draw = True

    elif event == cv2.EVENT_MOUSEMOVE and inputMode:
        if draw:
            frame = cv2.rectangle(copy, (ix, iy), (x, y), (255, 0, 0), 2)
            copy = orig.copy()

    elif event == cv2.EVENT_LBUTTONUP and inputMode:

        if draw:
            x1 = max(x, ix)
            y1 = max(y, iy)
            ix = min(x, ix)
            iy = min(y, iy)
            roi = orig[iy:y1,ix:x1]
        draw = False

def resize():
    global frame
    r = 400.0 / frame.shape[0]
    dim = (int(frame.shape[1] * r), 400)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def getLimits():
    global limits,roi,frame

    cv2.namedWindow('input')
    cv2.setMouseCallback('input', selectROI)
    while True:
        cv2.imshow('input',frame)
        if not(roi is None):
            roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(roi)
            limits = [(int(np.amax(h)),int(np.amax(s)),255),(int(np.amin(h)),int(np.amin(s)),int(np.amin(v)))]

            print limits
            cv2.destroyWindow('input')
            break

        k = cv2.waitKey(1) & 0xFF
        if k==ord('q'):
            cv2.destroyWindow('input')
            break


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
            inputMode = False
            upper = limits[0]
            lower = limits[1]

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            cv2.imshow('mask',mask)

            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None
            if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # only proceed if the radius meets a minimum size
                if radius > 5:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                               (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

        cv2.imshow('frame',frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord('i') and limits is None:
            inputMode = True
            getLimits()

    camera.release()
    cv2.destroyAllWindows()
