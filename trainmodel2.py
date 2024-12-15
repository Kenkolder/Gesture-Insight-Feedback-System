import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Gesture categories
gestures = ['okay', 'peace', 'call_me', 'fist', 'smile']

# Data collection arrays
data = []
labels = []

# Loop through each gesture folder
for idx, gesture in enumerate(gestures):
    gesture_path = f'data/{gesture}'
    
    # Make sure the directory exists
    if os.path.exists(gesture_path):
        for img_name in os.listdir(gesture_path):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(gesture_path, img_name)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue

                # Convert image to RGB for MediaPipe
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Process the image and get hand landmarks
                result = hands.process(img_rgb)
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Extract hand landmarks as a flat array
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])  # Collect x, y, z coordinates
                        
                        # Append the data and corresponding labels
                        data.append(landmarks)
                        labels.append(idx)
    else:
        print(f"Gesture path {gesture_path} does not exist.")

# Convert data and labels into NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Split the dataset into training and validation sets
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define a simple neural network model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(data.shape[1],)),  # Input size is the number of landmarks
    layers.Dense(64, activation='relu'),
    layers.Dense(len(gestures), activation='softmax')  # 5 gestures, so the output layer has 5 neurons
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=20, batch_size=32, validation_data=(test_data, test_labels))

# Save the model in .h5 format
model.save('hand_gesture_model.h5')

print("Model has been saved as hand_gesture_model.h5")
