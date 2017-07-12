import cv2
import numpy as np
import track_utils
from collections import deque
import math

frame = None
orig_frame = None
roi_hist_A, roi_hist_B = None, None
roi = None
<<<<<<< HEAD
team = None
=======
>>>>>>> b8a0bc709908e05faa88e3a204c624431dda394d
teamA = np.array([])
teamB = np.array([])
teamB_new = np.array([])
teamA_new = np.array([])
pts = []
<<<<<<< HEAD
minDist = 0
prevTeam = None
prevPasser = -1
=======
>>>>>>> b8a0bc709908e05faa88e3a204c624431dda394d
M = None
op = None
limits = None
ball_center = None

kernel = np.array([[0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0],
                   [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1],
                   [0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0]], dtype=np.uint8)

prevgrad = 0
passes = 0
<<<<<<< HEAD
vel, prev_vel = 0, 0
=======
vel, prev_vel = 0,0
>>>>>>> b8a0bc709908e05faa88e3a204c624431dda394d
pts_ball = deque()


def trackBall():
<<<<<<< HEAD
    global grad, prevgrad, passes, ball_center, pts_ball, frame, vel, prev_vel, prevPasser, prevTeam, minDist
    pts_ball.appendleft(ball_center)

    if len(pts_ball) > 2:
        if len(pts_ball) > 20:

            for i in xrange(1, 20):
                cv2.line(frame, pts_ball[i - 1], pts_ball[i], (0, 0, 255), 2, cv2.LINE_AA)
        else:
            for i in xrange(1, len(pts_ball)):
                cv2.line(frame, pts_ball[i - 1], pts_ball[i], (0, 0, 255), 2, cv2.LINE_AA)
=======
    global grad, prevgrad, passes, ball_center, pts_ball, frame, vel, prev_vel
    pts_ball.appendleft(ball_center)

    '''if len(pts_ball) > 2:

        for i in xrange(1, len(pts_ball)):
            cv2.line(frame, pts_ball[i-1], pts_ball[i], (0, 0, 255), 2, cv2.LINE_AA)
    '''
>>>>>>> b8a0bc709908e05faa88e3a204c624431dda394d

    l = len(pts_ball)
    if l >= 10:
        grad = np.arctan2((pts_ball[9][1] - pts_ball[0][1]), (pts_ball[9][0] - pts_ball[0][0]))
        grad = grad * (180.0 / np.pi)
        grad %= 360
<<<<<<< HEAD
        vel = math.sqrt((pts_ball[9][1] - pts_ball[0][1]) ** 2 + (pts_ball[9][0] - pts_ball[0][0]) ** 2) / 10
        if (math.fabs(grad - prevgrad) >= 20):
            # or math.fabs(vel-prev_vel) >= 7:
            # detectPlayers()
            # print("a " + str(len(teamA)) + "  b " + str(len(teamB)))
            if len(teamA) != 0 and len(teamB) != 0:
                getCoordinates()
                detectPasser()

                # print(passerIndex)

                if ((prevTeam != team) or (passerIndex != prevPasser)) and minDist < 10000:
                    # print(minDist)
                    # print(str(team) + str(passerIndex))
                    if (team == 'A'):

                        detectOffside()
                    else:
                        print('Not offside')

                    passes += 1
                    print('Ball Passed ' + str(passes))
                prevPasser = passerIndex
                prevTeam = team

=======
        vel = math.sqrt((pts_ball[9][1] - pts_ball[0][1]) ** 2 + (pts_ball[9][0] - pts_ball[0][0]) ** 2)/10
        if (math.fabs(grad - prevgrad) >= 20) :
                #or math.fabs(vel-prev_vel) >= 7:
            print('Ball Passed ' + str(passes))
            passes += 1
>>>>>>> b8a0bc709908e05faa88e3a204c624431dda394d
        prevgrad = grad
        prev_vel = vel


<<<<<<< HEAD
=======

>>>>>>> b8a0bc709908e05faa88e3a204c624431dda394d
def detectPlayers():
    global frame, roi_hist_A, roi_hist_B, teamA, teamB
    teamA = []
    teamB = []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cnt_thresh = 180
    if roi_hist_A is not None:
        backProjA = cv2.calcBackProject([hsv], [0, 1], roi_hist_A, [0, 180, 0, 256], 1)
        maskA = track_utils.applyMorphTransforms2(backProjA)
        cv2.imshow('mask a', maskA)

        cnts = cv2.findContours(maskA.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts) > 0:
            c = sorted(cnts, key=cv2.contourArea, reverse=True)
            for i in range(len(c)):
                if cv2.contourArea(c[i]) < cnt_thresh:
                    break

                x, y, w, h = cv2.boundingRect(c[i])
                h += 5
                y -= 5
                if h < 0.8 * w:
                    continue
                elif h / float(w) > 3:
                    continue

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                M = cv2.moments(c[i])
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                foot = (center[0], int(center[1] + h * 1.5))
                teamA.append(foot)
                cv2.circle(frame, foot, 5, (0, 0, 255), -1)
    if roi_hist_B is not None:
        backProjB = cv2.calcBackProject([hsv], [0, 1], roi_hist_B, [0, 180, 0, 256], 1)
        maskB = track_utils.applyMorphTransforms2(backProjB)
        cv2.imshow('mask b', maskB)

        cnts = cv2.findContours(maskB.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts) > 0:
            c = sorted(cnts, key=cv2.contourArea, reverse=True)
            for i in range(len(c)):
                if cv2.contourArea(c[i]) < cnt_thresh:
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
<<<<<<< HEAD
                foot = (center[0], int(center[1] + h * 1.2))
=======
                foot = (center[0], int(center[1] + h * 1.5))
>>>>>>> b8a0bc709908e05faa88e3a204c624431dda394d
                teamB.append(foot)
                cv2.circle(frame, foot, 5, (0, 0, 255), -1)


def selectPoints(event, x, y, flag, param):
    global pts, frame, orig_frame

    if event == cv2.EVENT_LBUTTONUP:
        if len(pts) < 8:
            pts.append([x, y])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        else:
            print('You have already selected 4 points')


def getBoundaryPoints():
    global frame, pts
    end_pts = []
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', selectPoints)
    while True:
        cv2.imshow('input', frame)
        key = cv2.waitKey(1) & 0xFF
        if len(pts) >= 8:
            pts = np.array(pts, dtype=np.float32)
            pts[:, 1] *= (-1)
            for i in range(0, 5, 2):
                m1 = (pts[i + 1][1] - pts[i][1]) / (pts[i + 1][0] - pts[i][0])
                m2 = (pts[i + 3][1] - pts[i + 2][1]) / (pts[i + 3][0] - pts[i + 2][0])
                A = np.array([[m1, -1], [m2, -1]])
                A_inv = np.linalg.inv(A)
                B = np.array([pts[i][1] - m1 * pts[i][0], pts[i + 2][1] - m2 * pts[i + 2][0]])
                B *= (-1)
                p = np.dot(A_inv, B)
                end_pts.append(np.int16(p))
            m1 = (pts[7][1] - pts[6][1]) / (pts[7][0] - pts[6][0])
            m2 = (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])
            A = np.array([[m1, -1], [m2, -1]])
            A_inv = np.linalg.inv(A)
            B = np.array([pts[6][1] - m1 * pts[6][0], pts[0][1] - m2 * pts[0][0]])
            B *= (-1)
            p = np.dot(A_inv, B)
            end_pts.append(np.int16(p))
            end_pts = np.array(end_pts)
            end_pts[:, 1] *= (-1)
            break
        elif key == ord("q"):
            break
    cv2.destroyWindow('input')
    return end_pts


def getCoordinates():
<<<<<<< HEAD
    global M, teamA, teamB, op, teamB_new, teamA_new, ball_new, ball_center
    teamB_new = np.array([])
    teamA_new = np.array([])
    op = orig_op.copy()
    if ball_center is not None:
        new = np.dot(M, [ball_center[0], ball_center[1], 1])
        ball_new = [new[0] / new[2], new[1] / new[2]]
        op = cv2.circle(op, (int(ball_new[0]), int(ball_new[1])), 3, (255, 0, 0), -1)
=======
    global M, teamA, teamB, op, teamB_new, teamA_new
    teamB_new = np.array([])
    teamA_new = np.array([])
>>>>>>> b8a0bc709908e05faa88e3a204c624431dda394d
    if len(teamB) > 0:
        for i in range(len(teamB)):
            new_pt = np.dot(M, [teamB[i][0], teamB[i][1], 1])
            teamB_new = np.append(teamB_new, [new_pt[0] / new_pt[2], new_pt[1] / new_pt[2]])
<<<<<<< HEAD
        teamB_new = np.int16(teamB_new).reshape(-1, 2)
        for i in range(len(teamB)):
            op = cv2.circle(op, (teamB_new[i][0], teamB_new[i][1]), 5, (0, 255, 0), -1)
=======
        teamB_new = np.int16(teamB_new).reshape(-1,2)
        for i in range(len(teamB)):
            op = cv2.circle(orig_op.copy(), (teamB_new[i][0], teamB_new[i][1]), 5, (0, 255, 0), -1)
>>>>>>> b8a0bc709908e05faa88e3a204c624431dda394d
    if len(teamA) > 0:
        for i in range(len(teamA)):
            new_pt = np.dot(M, [teamA[i][0], teamA[i][1], 1])
            teamA_new = np.append(teamA_new, [new_pt[0] / new_pt[2], new_pt[1] / new_pt[2]])
<<<<<<< HEAD
        teamA_new = np.int16(teamA_new).reshape(-1, 2)
        for i in range(len(teamA)):
            op = cv2.circle(op, (teamA_new[i][0], teamA_new[i][1]), 5, (0, 0, 255), -1)


def drawOffsideLine():
    global M, teamB_new, op, frame
    if len(teamB_new) > 0:
        M_inv = np.linalg.inv(M)
        p1 = np.dot(M_inv, [teamB_new[0][0], 0, 1])
        p2 = np.dot(M_inv, [teamB_new[0][0], op.shape[0] - 1, 1])

        pts = [(int(p1[0] / p1[2]), int(p1[1] / p1[2])), (int(p2[0] / p2[2]), int(p2[1] / p2[2]))]
        frame = cv2.line(frame, pts[0], pts[1], (255, 0, 0), 2)


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    node = np.array([node[0], node[1]])
    # print(nodes)
    # print(node)
    dist_2 = np.sum((nodes - node) ** 2, axis=1)

    return np.argmin(dist_2)


def detectPasser():
    global ball_new, teamA, teamB, passerIndex, team, minDist
    teamA_min_ind = closest_node(ball_new, teamA_new)
    teamB_min_ind = closest_node(ball_new, teamB_new)
    # print(np.asarray(teamA[teamA_min_ind]))
    # print(np.asarray(ball_center))
    teamA_min = np.sum(([np.asarray(teamA_new[teamA_min_ind])] - np.asarray(ball_new)) ** 2, axis=1)
    teamB_min = np.sum(([np.asarray(teamB_new[teamB_min_ind])] - np.asarray(ball_new)) ** 2, axis=1)
    minDist = min(teamB_min, teamA_min)
    if (teamA_min < teamB_min):
        # print("Ball passed by TeamA player")

        passerIndex = teamA_min_ind
        # print(passerIndex)
        team = 'A'
    else:
        # print("Ball passed by TeamB player")
        passerIndex = teamB_min_ind
        team = 'B'


def detectOffside():
    global teamA_new, teamB_new, passerIndex
    if len(teamB_new) > 0:
        if len(teamA_new) > 0:
            # teamA_new.sort()
            teamB_new.sort()
            # print(teamA_new)
            if (teamB_new[0][0] > teamA_new[passerIndex][0]):
                # if (teamB[0][0] > teamA[passerIndex][0]):
                # print(passerIndex)#Assuming no goalie
                print('Offside')
            else:
                print('Not Offside')
        else:
            print('Not Offside')
    else:
        print('Not Offside')
=======
        teamA_new = np.int16(teamA_new).reshape(-1,2)
        for i in range(len(teamA)):
            op = cv2.circle(op, (teamA_new[i][0], teamA_new[i][1]), 5, (0, 0, 255), -1)

def drawOffsideLine():
    global M, teamB_new, op, frame
    if len(teamB_new) >0:
        M_inv = np.linalg.inv(M)
        p1 = np.dot(M_inv, [teamB_new[0][0], 0, 1])
        p2 = np.dot(M_inv, [teamB_new[0][0], op.shape[0]-1, 1])

        pts = [(int(p1[0]/p1[2]), int(p1[1]/p1[2])),(int(p2[0]/p2[2]), int(p2[1]/p2[2]))]
        frame = cv2.line(frame, pts[0],pts[1],(255,0,0),2)
>>>>>>> b8a0bc709908e05faa88e3a204c624431dda394d


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

        frame = track_utils.resize(frame, width=400)
        orig_frame = frame.copy()
        frame2 = track_utils.removeBG(orig_frame.copy(), fgbg)

        detectPlayers()

        if roi is not None:
            ball_center, cnt = track_utils.detectBallThresh(frame2, limits)
            if cnt is not None:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(frame, (int(x), int(y)), int(radius), (255, 255, 0), 2)
                cv2.circle(frame, ball_center, 2, (0, 0, 255), -1)
                trackBall()

        if M is not None:
            src = np.int32(src)
<<<<<<< HEAD
            # print src
            for i in range(4):
                frame = cv2.circle(frame.copy(), (src[i][0], src[i][1]), 5, (255, 0, 255), -1)
            # frame = cv2.polylines(frame.copy(), src,True,(255,0,255),3)
=======
            #print src
            for i in range(4):
                frame = cv2.circle(frame.copy(), (src[i][0],src[i][1]), 5, (255, 0, 255), -1)
            #frame = cv2.polylines(frame.copy(), src,True,(255,0,255),3)
>>>>>>> b8a0bc709908e05faa88e3a204c624431dda394d
            getCoordinates()
            drawOffsideLine()

        cv2.imshow('frame', frame)
        cv2.imshow('output', op)
        if flag:
            t = 1
        else:
            t = 10

        key = cv2.waitKey(t) & 0xFF

        if key == ord("q"):
            break
        elif key == ord('i') and (roi_hist_A is None or roi_hist_B is None):
            flag = True
            roi_hist_A, roi_hist_B = track_utils.getHist(frame)

            roi = track_utils.getROIvid(orig_frame, 'input ball')
            if roi is not None:
                limits = track_utils.getLimits(roi)

            src = getBoundaryPoints()
            src = np.float32(src)
            dst = np.float32([[0, 0], [0, op.shape[0]], [op.shape[1], op.shape[0]], [op.shape[1], 0]])
            M = cv2.getPerspectiveTransform(src, dst)

    camera.release()
    cv2.destroyAllWindows()
