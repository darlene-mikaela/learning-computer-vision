import cv2
import mediapipe as mp

vid = cv2.VideoCapture(0)
vid.set(3, 960)
mphands = mp.solutions.hands
Hands = mphands.Hands(max_num_hands= 1, min_detection_confidence= 0.7, min_tracking_confidence= 0.6)

def openedOrNot(positions, handedness):
    if ((positions[8][1] > positions[6][1])
        and (positions[12][1] > positions[10][1])
        and (positions[16][1] > positions[14][1])
        and (positions[20][1] > positions[18][1])):

        if (handedness=="Right" and positions[4][0] > positions[6][0]):
            return "FULLY CLOSED"
        elif (handedness=="Left" and positions[4][0] < positions[6][0]):
            return "FULLY CLOSED"
        else:
            return "NOT FULLY CLOSED"
    else:
        return "NOT FULLY CLOSED"


while vid.isOpened():
    success, frame = vid.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # convert from bgr to rgb
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = Hands.process(RGBframe)  
    mpdraw = mp.solutions.drawing_utils 
    handedness_label = True

    #handedness (l/r)
    if result.multi_handedness:
        for hand_info in result.multi_handedness:
            handedness_label = hand_info.classification[0].label
            cv2.putText(frame, handedness_label, (850, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4) 
    
    if result.multi_hand_landmarks:
        for handLm in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            points = {}

            for id, lm in enumerate(handLm.landmark):
                cx, cy = int(lm.x*w), int(lm.y*h)
                points[id] = (cx, cy) #create a list of (x,y) coordinates for each point
                mpdraw.draw_landmarks(frame, handLm, mphands.HAND_CONNECTIONS)

            handStatus = openedOrNot(points, handedness_label)
            cv2.putText(frame, handStatus, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
    
    cv2.imshow("video", frame)
    cv2.waitKey(1)
