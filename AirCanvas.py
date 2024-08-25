import mediapipe as mp
import cv2
import numpy as np


"""
step1 - connect with webcam   - done
step2 - detect hand           - done
step3 - configure fingers     - done
step4 - write in the air
"""

# Initialising mediapipe hands
mphands = mp.solutions.hands
hands = mphands.Hands(
    min_detection_confidence = 0.9
)

# for drawing the joints 
mpdraw = mp.solutions.drawing_utils

# Initialising a capture device for webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # the second param is for fast opening of the webcam

canvas = np.zeros((480, 640, 3), np.uint8)

drawColor = (255, 255, 52)#silver(192, 192, 192) #BGR as the canvas is flipped
BrushThickness = 10
xp, yp = 0, 0

def finger_state(tip, joint):
    # If tip is below the joint than the finger is curled
    return 1 if tip.y < joint.y else 0

# Only for right thumb
def thumb_state(tip, joint, hand_label):
    if hand_label == 'Right':
        return 1 if tip.x < joint.x else 0
    else:
        return 1 if tip.x > joint.x else 0

# Looping so that the webcam does not shut down prematurely 
while cap.isOpened():
    success, img = cap.read() 
    if not success:
        print("Device not found!")
    else:
        # height, width, channel = img.shape
        # print(height, width, channel)

        # Converting from bgr to rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Processing the hands in the normal img (without flipping)
        results = hands.process(img_rgb)
        # print(results.multi_hand_landmarks) 

        # Detecting the hands in the frame
        if results.multi_hand_landmarks and results.multi_handedness:
            for i, hand_marks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[i].classification[0].label

                index_open = finger_state(hand_marks.landmark[8], hand_marks.landmark[6]) == 1
                
                all_open = (
                 finger_state(hand_marks.landmark[8], hand_marks.landmark[6]) == 1 and #Index finger
                 finger_state(hand_marks.landmark[12], hand_marks.landmark[10]) == 1 and #middle finger
                 finger_state(hand_marks.landmark[16], hand_marks.landmark[14]) == 1 and#ring finger
                 finger_state(hand_marks.landmark[20], hand_marks.landmark[18])) #pinky 

                #  Drawing the hand landmarks on the hand
                mpdraw.draw_landmarks(img, hand_marks, mphands.HAND_CONNECTIONS)

                index_tip = hand_marks.landmark[8]
                h, w, c = img.shape
                x1, y1 = int(index_tip.x*w), int(index_tip.y*h)
                
                if index_open and not all_open:
                    cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)
                    # print('Index finger open')
                    if xp ==0 and yp == 0 :
                        xp, yp = x1, y1
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, BrushThickness)
                    cv2.line(canvas, (xp, yp), (x1, y1), drawColor, BrushThickness)

                    xp, yp = x1, y1

                else:
                    xp, yp = 0,0

                # if all_open:
                    # print('All fingers are opened')

        
        imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgINV = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgINV = cv2.cvtColor(imgINV, cv2.COLOR_GRAY2BGR)

        img = cv2.bitwise_and(img, imgINV)
        img = cv2.bitwise_or(img, canvas)
        # Starting the webcam
        # img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
        cv2.imshow('Air Canvas', cv2.flip(img, 1))
        cv2.imshow(' Canvas', cv2.flip(canvas, 1))

    # If Esc is pressed the webcam window will be closed
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows( )