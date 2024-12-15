import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Gesture list
gestures = ['okay', 'peace', 'call_me', 'fist', 'smile']

# Create dataset
data = []
labels = []

# Loop over each gesture
for idx, gesture in enumerate(gestures):
    gesture_path = f'data/{gesture}'
    
    if not os.path.exists(gesture_path):
        print(f"Directory {gesture_path} does not exist. Skipping...")
        continue

    for img_name in os.listdir(gesture_path):
        img_path = os.path.join(gesture_path, img_name)
        
        # Try to read image and handle errors
        img = cv2.imread(img_path)
        if img is None:
            print(f"Unable to read {img_path}. Skipping this file...")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image for hand landmarks
        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                
                # Flatten landmarks and append to data
                landmarks = np.array(landmarks).flatten()
                data.append(landmarks)
                labels.append(idx)
        else:
            print(f"No hand landmarks detected in {img_name}.")

# Convert to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Save the dataset
np.save('hand_gesture_data.npy', data)
np.save('hand_gesture_labels.npy', labels)

print(f"Dataset created with {len(data)} samples.")
