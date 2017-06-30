import cv2
import numpy as np
import argparse

frame = None
orig = None
roi = None
roi_hist_A, roi_hist_B = None, None

ix, iy = 0, 0
teamA_lim, teamB_lim = None, None
draw = False
center = None

def getArguements():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    args = vars(ap.parse_args())
    return args

def resize(width):
    global frame
    r = float(width) / frame.shape[0]
    dim = (int(frame.shape[1] * r), width)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def selectROI(event, x, y, flag, param):
    global roi, frame, orig, ix, iy, draw
    copy =orig.copy()
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
        frame = orig
        draw = False

def getLimits():
    global teamA_lim,teamB_lim, roi, roi_hist_A, roi_hist_B, frame

    if teamA_lim is None:
        winName = 'input team A'
        cv2.namedWindow(winName)
        cv2.setMouseCallback(winName, selectROI)
        while True:
            cv2.imshow(winName, frame)
            if roi is not None:
                roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
                roi_hist_A = cv2.calcHist([roi],[0,1],None,[180,256],[0,180,0,256])
                roi_hist_A = cv2.normalize(roi_hist_A, roi_hist_A, 0, 255, cv2.NORM_MINMAX)

                cv2.destroyWindow(winName)
                roi = None
                break

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                cv2.destroyWindow(winName)
                break

    if teamB_lim is None:
        winName = 'input team B'
        cv2.namedWindow(winName)
        cv2.setMouseCallback(winName, selectROI)
        while True:
            cv2.imshow(winName, frame)
            if roi is not None:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                roi_hist_B = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
                roi_hist_B = cv2.normalize(roi_hist_B,roi_hist_B, 0, 255, cv2.NORM_MINMAX)

                cv2.destroyWindow(winName)
                roi = None
                break

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                cv2.destroyWindow(winName)
                break

def detectPlayers():
    global center,teamB_lim,teamA_lim,frame, roi_hist_A, roi_hist_B

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = 0.1
    upper = 1
    cnt_thresh = 300
    if roi_hist_A is not None:
        maskA = cv2.calcBackProject([hsv], [0,1], roi_hist_A, [0, 180, 0, 256], 1)
        maskA = cv2.inRange(maskA, lower, upper)
        maskA = cv2.bitwise_not(maskA)
        maskA = cv2.dilate(maskA, np.ones((5,5)), iterations=2)
        maskA = cv2.erode(maskA, np.ones((3, 3)))

        cv2.imshow('mask a',maskA)

        cnts = cv2.findContours(maskA.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts) > 3:
            c = sorted(cnts,key=cv2.contourArea,reverse=True)
            for i in range(len(c)):
                if cv2.contourArea(c[i])<cnt_thresh:
                    break

                x, y, w, h = cv2.boundingRect(c[i])

                if h < 1.1 * w:
                    continue
                elif h / float(w) > 3:
                    continue

                frame = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
                M = cv2.moments(c[i])
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

    if roi_hist_B is not None:
        maskB = cv2.calcBackProject([hsv], [0, 1], roi_hist_B, [0, 180, 0, 256], 1)
        maskB = cv2.inRange(maskB, lower, upper)
        maskB = cv2.bitwise_not(maskB)
        maskB = cv2.dilate(maskB, np.ones((5,5)), iterations=2)
        maskB = cv2.erode(maskB, np.ones((3, 3)))

        cv2.imshow('mask b',maskB)

        cnts = cv2.findContours(maskB.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts) > 3:
            c = sorted(cnts,key=cv2.contourArea,reverse=True)
            for i in range(len(c)):
                if cv2.contourArea(c[i])<cnt_thresh:
                    break
                x, y, w, h = cv2.boundingRect(c[i])

                if h < 1.1 * w:
                    continue
                elif h / float(w) > 3:
                    continue

                frame = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
                M = cv2.moments(c[i])
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(frame, center, 5, (0, 0, 255), -1)


if __name__ == '__main__':
    args = getArguements()

    if not args.get("video", False):
        camera = cv2.VideoCapture(1)
    else:
        camera = cv2.VideoCapture(args["video"])

    while True:
        (grabbed, frame) = camera.read()

        if args.get("video") and not grabbed:
            break

        resize(width=400)
        frame = cv2.bilateralFilter(frame, 5, 15, 15)
        orig = frame.copy()

        detectPlayers()
        cv2.imshow('frame', orig)


        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break
        elif key == ord('i') and (teamA_lim is None or teamB_lim is None):
            getLimits()

    camera.release()
    cv2.destroyAllWindows()