import cv2
import numpy as np
import track_utils

frame = None
orig_frame = None
pts = []
p = None
inp = False
M = None

def selectPoints(event, x, y, flag, param):

    global pts, frame, orig_frame

    if event == cv2.EVENT_LBUTTONUP:
        if len(pts) < 8:
            pts.append([x, y])
            cv2.circle(frame, (x,y), 5, (0, 0, 255), -1)
            print pts
        else:
            print('You have already selected 4 points')

def getPoint(event,x,y,flag,param):
    global frame,p,orig_frame
    if event == cv2.EVENT_MOUSEMOVE:
        p = (x,y)
        frame = cv2.circle(orig_frame.copy(),p,5,(0,0,255),-1)

def getBoundaryPoints():
    global frame, pts
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', selectPoints)
    end_pts = []

    while True:
        cv2.imshow('input', frame)
        key = cv2.waitKey(1) & 0xFF

        if len(pts) >= 8:
            pts = np.array(pts)
            pts *= (-1)
            for i in range(0,5,2):
                m1 = (pts[i+1][1]-pts[i][1])/(pts[i+1][0]-pts[i][0])
                m2 = (pts[i+3][1]-pts[i+2][1])/(pts[i+3][0]-pts[i+2][0])
                A = np.array([[m1, -1], [m2, -1]])
                A_inv = np.linalg.inv(A)
                B = np.array([pts[i][1] - m1*pts[i][0], pts[i+2][1] - m2*pts[i+2][0]])
                p = np.dot(A_inv,B)
                print p
                end_pts.append(np.int16(p))
            m1 = (pts[7][1] - pts[6][1]) / (pts[7][0] - pts[6][0])
            m2 = (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])
            A = np.array([[m1, -1], [m2, -1]])
            A_inv = np.linalg.inv(A)
            B = np.array([pts[6][1] - m1 * pts[6][0], pts[0][1] - m2 * pts[0][0]])
            p = np.dot(A_inv, B)
            print p
            end_pts.append(np.int16(p))
            print (end_pts)

            break
        elif key == ord("q"):
            break
    cv2.destroyWindow('input')
    return end_pts

def getCoordinates():
    global M,p,op
    if p is not None:
        new_pt = np.dot(M,[p[0],p[1],1])
        teamB_new = np.array([new_pt[0]/new_pt[2],new_pt[1]/new_pt[2]],dtype=np.uint16)
        op = cv2.circle(orig_op.copy(),(teamB_new[0],teamB_new[1]),5,(0,0,255),-1)


if __name__ == '__main__':

    orig_op = cv2.imread('soccer_half_field.jpeg')
    op = orig_op.copy()

    '''args = track_utils.getArguements()

    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args["video"])
        '''
    frame = cv2.imread('soccer_field.jpg')
    orig_frame = frame.copy()
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame',getPoint)
    while True:
        #_, frame = camera.read()
        #orig_frame = frame.copy()
        if M is not None:
            getCoordinates()
        cv2.imshow('frame',frame)
        cv2.imshow('output', op)

        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
        elif k == ord('i'):
            inp = True
            src = getBoundaryPoints()
            src = np.float32(src)
            dst = np.float32([[0, 0], [0, op.shape[0]], [op.shape[1], op.shape[0]], [op.shape[1], 0]])
            p2 = np.array([[0, 0], [0, op.shape[0]], [op.shape[1], op.shape[0]], [op.shape[1], 0]],dtype=np.int32)

            cv2.polylines(orig_op,[p2],True,(255,0,0),2)
            print dst
            print op.shape

            M = cv2.getPerspectiveTransform(src,dst)
            inp = False
            print M


    #camera.release()
    cv2.destroyAllWindows()