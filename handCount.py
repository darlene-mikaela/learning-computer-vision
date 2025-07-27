import cv2
import mediapipe as mp

vid = cv2.VideoCapture(0)
vid.set(3, 960)
mphands = mp.solutions.hands
Hands = mphands.Hands(max_num_hands= 1, min_detection_confidence= 0.7, min_tracking_confidence= 0.6)

def handOpenStatus(positions, handedness):
    fourFingersClosed = (positions[8][1] > positions[6][1]) and (positions[12][1] > positions[10][1]) and (positions[16][1] > positions[14][1]) and (positions[20][1] > positions[18][1])
    fourFingersOpened = (positions[8][1] < positions[6][1]) and (positions[12][1] < positions[10][1]) and (positions[16][1] < positions[14][1]) and (positions[20][1] < positions[18][1])

    if (handedness=="Right"):
        if fourFingersOpened:
            return "FULLY OPENED" if positions[4][0] < positions[6][0] else "THUMB IN"
        elif fourFingersClosed:
            return "FULLY CLOSED" if positions[4][0] > positions[6][0] else "THUMB OUT"
        else:
            return "NOT OPENED/CLOSED"
    else:
        if fourFingersOpened:
            return "FULLY OPENED" if positions[4][0] > positions[6][0] else "THUMB IN"
        elif fourFingersClosed:
            return "FULLY CLOSED" if positions[4][0] < positions[6][0] else "THUMB OUT"
        else:
            return "NOT OPENED/CLOSED"

def handCountLeft(positions):
    if (positions[8][1] > positions[6][1]
        and positions[12][1] > positions[10][1]
        and positions[16][1] > positions[14][1]
        and positions[20][1] > positions[18][1]
        and positions[4][0] < positions[6][0]):
        return "0"
    elif (positions[8][1] < positions[6][1]
        and positions[12][1] > positions[11][1]
        and positions[16][1] > positions[14][1]
        and positions[20][1] > positions[18][1]
        and positions[4][0] < positions[6][0]):
        return "1"
    elif (positions[8][1] < positions[6][1]
        and positions[12][1] < positions[11][1]
        and positions[16][1] > positions[14][1]
        and positions[20][1] > positions[18][1]
        and positions[4][0] < positions[6][0]):
        return "2"
    elif (positions[8][1] < positions[6][1]
        and positions[12][1] < positions[11][1]
        and positions[16][1] < positions[14][1]
        and positions[20][1] > positions[18][1]
        and positions[4][0] < positions[6][0]):
        return "3"
    elif (positions[8][1] < positions[6][1]
        and positions[12][1] < positions[11][1]
        and positions[16][1] < positions[14][1]
        and positions[20][1] < positions[18][1]
        and positions[4][0] < positions[6][0]):
        return "4"
    # need todo 
    elif (positions[8][1] < positions[6][1]
        and positions[12][1] > positions[11][1]
        and positions[16][1] > positions[14][1]
        and positions[20][1] > positions[18][1]
        and positions[4][0] > positions[6][0]):
        return "5"
    elif ((positions[8][1] > positions[6][1]
        and positions[12][1] > positions[11][1]
        and positions[16][1] > positions[14][1]
        and positions[20][1] > positions[18][1]
        and positions[4][0] > positions[6][0]) or
        (positions[8][0] < positions[6][0]
        and positions[12][0] < positions[11][0]
        and positions[16][0] < positions[14][0]
        and positions[20][0] < positions[18][0]
        and positions[4][1] > positions[6][1])):
        return "6"
    elif (positions[8][1] < positions[6][1]
        and positions[12][1] > positions[11][1]
        and positions[16][1] > positions[14][1]
        and positions[20][1] > positions[18][1]
        and positions[4][0] > positions[6][0]):
        return "7"
    else:
        return "???"

def handCountRight(positions):
    if (positions[8][1] > positions[6][1]
        and positions[12][1] > positions[10][1]
        and positions[16][1] > positions[14][1]
        and positions[20][1] > positions[18][1]
        and positions[4][0] > positions[6][0]):
        return "0"
    elif (positions[8][1] < positions[6][1]
        and positions[12][1] > positions[11][1]
        and positions[16][1] > positions[14][1]
        and positions[20][1] > positions[18][1]
        and positions[4][0] > positions[6][0]):
        return "1"
    else:
        return "???"


while vid.isOpened():
    success, frame = vid.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # convert from bgr to rgb
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = Hands.process(RGBframe)  
    mpdraw = mp.solutions.drawing_utils 

    #handedness (l/r)
    if result.multi_handedness:
        for hand_info in result.multi_handedness:
            handedness = hand_info.classification[0].label
            cv2.putText(frame, handedness, (850, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4) 
    
    if result.multi_hand_landmarks:
        for handLm in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            points = {}

            # draw box over hand
            x_coords = [lm.x*w for lm in handLm.landmark]
            y_coords = [lm.y*h for lm in handLm.landmark]
            minX, maxX = int(min(x_coords)), int(max(x_coords))
            minY, maxY = int(min(y_coords)), int(max(y_coords))
            cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 255, 0), 2)

            for id, lm in enumerate(handLm.landmark):
                cx, cy = int(lm.x*w), int(lm.y*h)
                points[id] = (cx, cy) #create a list of (x,y) coordinates for each point
                mpdraw.draw_landmarks(frame, handLm, mphands.HAND_CONNECTIONS)

            handCountResult = handCountLeft(points) if handedness=="Left" else handCountRight(points)
            cv2.putText(frame, handCountResult, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
    
    cv2.imshow("video", frame)
    cv2.waitKey(1)
