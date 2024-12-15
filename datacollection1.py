import cv2
import mediapipe as mp
import os

# Create directory structure
gestures = ['okay', 'peace', 'call_me', 'fist', 'smile']
for gesture in gestures:
    os.makedirs(f'data/{gesture}', exist_ok=True)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

gesture_id = 0  # Change this to collect images for different gestures

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get hand landmark prediction
    result = hands.process(rgb_frame)

    # Draw hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
    
    # Show frame
    cv2.imshow("Capture Gestures", frame)
    
    # Press 'c' to capture an image
    if cv2.waitKey(1) & 0xFF == ord('c'):
        img_name = f"data/{gestures[gesture_id]}/{len(os.listdir(f'data/{gestures[gesture_id]}'))}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Captured {img_name}")
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()