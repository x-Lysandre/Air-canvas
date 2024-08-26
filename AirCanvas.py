import mediapipe as mp
import cv2
import numpy as np

"""
Air Canvas Functionality:
Write: Use your right hand's index finger to write on the canvas.
Toggle Erase Mode: Pinch with your right hand's thumb and index finger to switch between writing and erasing.
Change Color: Use your left hand to change the pen color:
    Index Finger: Cyan
    Index + Middle Finger: Peach
    Index + Middle + Ring Finger: Lime
    Index + Middle + Ring + Pinky Finger: Silver
"""

# Initializing Mediapipe Hands
mphands = mp.solutions.hands
hands = mphands.Hands(min_detection_confidence=0.9)

# For drawing the joints
mpdraw = mp.solutions.drawing_utils

# Initializing a capture device for webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Fast opening of the webcam

# Canvas and drawing settings
canvas = np.zeros((480, 640, 3), np.uint8)
drawColor = (255, 255, 52)  # BGR color for drawing
BrushThickness = 10
xp, yp = 0, 0
# for erasing
EraserThickness = 50
Black_Color = (0, 0, 0)    # Color for erasing
erase_flag = 0
last_pinch_time = 0
debounce_time = 500

def distance(tip1, tip2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((tip1.x - tip2.x)**2 + (tip1.y - tip2.y)**2 + (tip1.z - tip2.z)**2)

def finger_state(tip, joint):
    """Check if the finger is open based on the tip and joint position."""
    return 1 if tip.y < joint.y else 0

def thumb_state(tip, joint, hand_label):
    """Check if the thumb is open based on hand label."""
    if hand_label == 'Right':
        return 1 if tip.x < joint.x else 0
    return 1 if tip.x > joint.x else 0

# Looping so that the webcam does not shut down prematurely
while cap.isOpened():
    success, img = cap.read() 
    if not success:
        print("Device not found!")
        break
    
    # Converting from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Processing the hands in the image
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_marks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[i].classification[0].label
            # print(f'{hand_label} hand detected')

            # Drawing the hand landmarks on the image
            mpdraw.draw_landmarks(img, hand_marks, mphands.HAND_CONNECTIONS)

            if hand_label == 'Left':#because of the flip left is detected as right

                index_open = finger_state(hand_marks.landmark[8], hand_marks.landmark[6]) == 1
                pinch_dis = distance(hand_marks.landmark[4], hand_marks.landmark[8])
                
                all_open = (
                    finger_state(hand_marks.landmark[8], hand_marks.landmark[6]) == 1 and  # Index finger
                    finger_state(hand_marks.landmark[12], hand_marks.landmark[10]) == 1 and # Middle finger
                    finger_state(hand_marks.landmark[16], hand_marks.landmark[14]) == 1 and # Ring finger
                    finger_state(hand_marks.landmark[20], hand_marks.landmark[18])  # Pinky
                )


                index_tip = hand_marks.landmark[8]
                h, w, c = img.shape
                x1, y1 = int(index_tip.x * w), int(index_tip.y * h)

                # print(erase_flag)
                
                if erase_flag == 1:
                    # ERASE MODE    
                    if all_open:
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        cv2.line(img, (xp, yp), (x1, y1), Black_Color, EraserThickness)
                        cv2.line(canvas, (xp, yp), (x1, y1), Black_Color, EraserThickness)
                        xp, yp = x1, y1
                        current_time = cv2.getTickCount() / cv2.getTickFrequency() * 1000  # Current time in milliseconds
                        if pinch_dis < 0.05 and (current_time - last_pinch_time) > debounce_time:
                            erase_flag = 1 - erase_flag
                            last_pinch_time = current_time
                        
                else:
                    # PEN MODE
                    if index_open and not all_open:
                        cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, BrushThickness)
                        cv2.line(canvas, (xp, yp), (x1, y1), drawColor, BrushThickness)
                        xp, yp = x1, y1
                    else:
                        xp, yp = 0, 0

                    # Check pinch gesture to toggle erase mode
                    current_time = cv2.getTickCount() / cv2.getTickFrequency() * 1000  # Current time in milliseconds
                    if pinch_dis < 0.05 and (current_time - last_pinch_time) > debounce_time:
                        erase_flag = 1 - erase_flag
                        last_pinch_time = current_time
            else:
                # print('hello world')

                index_open = finger_state(hand_marks.landmark[8], hand_marks.landmark[6]) == 1
                middle_open = finger_state(hand_marks.landmark[12], hand_marks.landmark[10]) == 1
                ring_open = finger_state(hand_marks.landmark[16], hand_marks.landmark[14]) == 1
                pinky_open = finger_state(hand_marks.landmark[20], hand_marks.landmark[18]) == 1

                if index_open and not middle_open and not ring_open and not pinky_open:
                    drawColor = (255,255,52) # cian
                
                if index_open and  middle_open and not ring_open and not pinky_open:
                    drawColor = (153,51,255) # peach

                if index_open and  middle_open and  ring_open and not pinky_open:
                    drawColor = (51,255,153) #lime

                if index_open and  middle_open and  ring_open and  pinky_open:
                    drawColor = (160,160,160) #silver


    # Update canvas display
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgINV = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgINV = cv2.cvtColor(imgINV, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgINV)
    img = cv2.bitwise_or(img, canvas)
    
    # Displaying the images
    cv2.imshow('Air Canvas', cv2.flip(img, 1))
    cv2.imshow('Canvas', cv2.flip(canvas, 1))

    # Exit on 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
