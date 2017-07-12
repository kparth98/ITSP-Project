import cv2
import numpy as np
import track_utils
from collections import deque
import math

frame = None
orig_frame = None
roi_hist_A, roi_hist_B = None, None
teamA = np.array([])
teamB = np.array([])
pts = []
M = None
op = None
limits = None

kernel = np.array([[0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0]],dtype=np.uint8)

prevgrad = 0
passes = 0
pts_ball = deque()
orig = None

def trackBall():
    global grad, prevgrad, passes, center, orig, pts_ball, frame
    pts_ball.appendleft(center)

    for i in xrange(1, len(pts)):
        cv2.line(frame, pts_ball[i-1], pts_ball[i], (0, 0, 255), 2, cv2.LINE_AA)

    l = len(pts)
    if l >= 10:
        grad = np.arctan2((pts_ball[9][1] - pts_ball[0][1]), (pts_ball[9][0] - pts_ball[0][0]))
        grad = grad * 180 / np.pi
        grad %= 360

        if(math.fabs(grad - prevgrad) >= 10) and math.sqrt((pts_ball[9][1] - pts_ball[0][1]) ** 2 + (pts_ball[9][0] - pts_ball[0][0]) ** 2) >= 10:
            print('Ball Passed ' + str(passes))
            passes += 1
        prevgrad = grad

def applyMorphTransforms(backProj):
    global kernel
    lower = 50
    upper = 255
    mask = cv2.inRange(backProj, lower, upper)
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, np.ones((3, 3)))
    mask = cv2.GaussianBlur(mask, (11, 11), 5)
    mask = cv2.inRange(mask, lower, upper)
    return mask

def detectPlayers():
    global frame, roi_hist_A, roi_hist_B, teamA, teamB
    teamA = np.array([])
    teamB = np.array([])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cnt_thresh = 180
    if roi_hist_A is not None:
        backProjA = cv2.calcBackProject([hsv], [0,1], roi_hist_A, [0, 180, 0, 256], 1)
        maskA = applyMorphTransforms(backProjA)
        cv2.imshow('mask a',maskA)

        cnts = cv2.findContours(maskA.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts) > 0:
            c = sorted(cnts,key=cv2.contourArea,reverse=True)
            for i in range(len(c)):
                if cv2.contourArea(c[i])<cnt_thresh:
                    break

                x, y, w, h = cv2.boundingRect(c[i])
                h += 5
                y -= 5
                if h < 0.8*w:
                    continue
                elif h / float(w) > 3:
                    continue

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                M = cv2.moments(c[i])
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                foot = (center[0], int(center[1] + h*1.5))
                teamA = np.append(teamA,foot)
                cv2.circle(frame, foot, 5, (0, 0, 255), -1)

    if roi_hist_B is not None:
        backProjB = cv2.calcBackProject([hsv], [0, 1], roi_hist_B, [0, 180, 0, 256], 1)
        maskB = applyMorphTransforms(backProjB)
        cv2.imshow('mask b',maskB)

        cnts = cv2.findContours(maskB.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts) > 0:
            c = sorted(cnts,key=cv2.contourArea,reverse=True)
            for i in range(len(c)):
                if cv2.contourArea(c[i])<cnt_thresh:
                    break
                x, y, w, h = cv2.boundingRect(c[i])
                h += 5
                y -= 5
                if h < 0.9 * w:
                    continue
                elif h / float(w) > 3:
                    continue

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                M = cv2.moments(c[i])
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                foot = (center[0], int(center[1] + h * 1.5))
                teamB = np.append(teamB, foot)
                cv2.circle(frame, foot, 5, (0, 0, 255), -1)


def selectPoints(event, x, y, flag, param):

    global pts, frame, orig_frame

    if event == cv2.EVENT_LBUTTONUP:
        if len(pts) < 4:
            pts.append([x, y])
            cv2.circle(frame, (x,y), 5, (0, 0, 255), -1)
            print pts
        else:
            print('You have already selected 4 points')


def getBoundaryPoints():
    global frame, pts
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', selectPoints)
    while True:
        cv2.imshow('input', frame)
        key = cv2.waitKey(1) & 0xFF
        if len(pts) >= 4:
            break
        elif key == ord("q"):
            break
    cv2.destroyWindow('input')
    return pts

def getCoordinates():
    global M,teamA,teamB,op
    if len(teamB)>0:
        new_pt = np.dot(M,[teamB[0],teamB[1],1])
        teamB_new = np.array([new_pt[0]/new_pt[2],new_pt[1]/new_pt[2]],dtype=np.uint16)
        op = cv2.circle(orig_op.copy(),(teamB_new[0],teamB_new[1]),5,(0,0,255),-1)


if __name__ == '__main__':
    args = track_utils.getArguements()

    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])

    orig_op = cv2.imread('soccer_half_field.jpeg')
    op = orig_op.copy()
    fgbg = cv2.createBackgroundSubtractorMOG2(history=20, detectShadows=False)
    flag = False

    while True:
        (grabbed, frame) = camera.read()

        if args.get("video") and not grabbed:
            break

        frame = track_utils.resize(frame,width=400)
        orig_frame = frame.copy()
        frame2 = track_utils.removeBG(frame.copy(), fgbg)

        if limits is not None:
            center,cnt = track_utils.detectBallThresh(frame2,limits)
            if cnt is not None:
                (x,y),radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(frame,(int(x),int(y)),int(radius),(255,255,0),2)
                cv2.circle(frame,center,2,(0,0,255),-1)
                trackBall()

        detectPlayers()

        #if M is not None:
         #   getCoordinates()

        cv2.imshow('frame', frame)
        cv2.imshow('output', op)
        if flag :
            t = 1
        else:
            t = 100

        key = cv2.waitKey(t) & 0xFF

        if key == ord("q"):
            break
        elif key == ord('i') and (roi_hist_A is None or roi_hist_B is None):
            flag = True
            roi_hist_A, roi_hist_B = track_utils.getHist(frame)

            roi = track_utils.getROIvid(orig_frame, 'input ball')
            if roi is not None:
                limits = track_utils.getLimits(roi)

            #src = getBoundaryPoints()
            #src = np.float32(src)
            #dst = np.float32([[0, 0], [0, op.shape[0]], [op.shape[1], op.shape[0]], [op.shape[1], 0]])
            #M = cv2.getPerspectiveTransform(src, dst)

            print M
        #elif key == ord('s'):

    camera.release()
    cv2.destroyAllWindows()