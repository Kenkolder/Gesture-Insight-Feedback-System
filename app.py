import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
import logging

# Suppress TensorFlow logs
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Load the trained model
model = tf.keras.models.load_model('hand_gesture_model.h5')

# Define the gesture map
gesture_map = {
    0: 'GUI/UX Issues',  # okay
    1: 'Security Vulnerabilities',  # peace
    2: 'Performance Bottlenecks',  # call me
    3: 'Compatibility Issues',  # fist
    4: 'Functionality Bugs'   # smile
}

class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FEEDBACK APPLICATION")
        self.root.configure(bg="#D3D3D3")  # background

        # Configure the grid layout for dividing the screen
        self.root.grid_columnconfigure(0, weight=1)  # Left side
        self.root.grid_columnconfigure(1, weight=0)  # Middle column for divider
        self.root.grid_columnconfigure(2, weight=2)  # Right side
        self.root.grid_rowconfigure(0, weight=1)  # Single row

        # Left-side frame for title, text, image, and feedback points
        self.left_frame = tk.Frame(self.root, bg="#D3D3D3")
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        # Add title
        self.title_label = Label(self.left_frame, text="HAND GESTURE FEEDBACK SYSTEM", font=("Playfair Display", 35, "bold"), bg="#D3D3D3", fg="black")
        self.title_label.pack(pady=20)

        # Add descriptive text
        self.description_label = Label(self.left_frame, text="Real-time hand gesture recognition for feedback submissions.",
                                        font=("Arial", 18), bg="#D3D3D3", wraplength=500, justify="left")
        self.description_label.pack(pady=20)

        # Bullet points
        self.bullets_frame = tk.Frame(self.left_frame, bg="#D3D3D3")
        self.bullets_frame.pack(pady=5)

        feedback_points = [
            ("1. GUI/UX Issues", [
                "* Inconsistent Font and Colors: Fonts or colors vary unnecessarily across pages.",
                "* Cluttered Layout: Too many elements on a single screen, overwhelming the user.",
                "* Poor Accessibility: Lack of screen reader support or inadequate color contrast for visually impaired users.",
                "* Slow Feedback on Actions: Buttons or features lack immediate visual or functional response."
            ]),
            ("2. Security Vulnerabilities", [
                "* Weak Authentication: Missing features like two-factor authentication (2FA) or poor password strength validation.",
                "* Unencrypted Data Storage: Sensitive information stored without encryption.",
                "* Insecure APIs: Endpoints exposed to unauthorized access due to lack of token-based authentication.",
                "* Data Leakage: Sensitive data exposed in logs, error messages, or URLs."
            ]),
            ("3. Performance Bottlenecks", [
                "* Slow Database Queries: Queries taking too long due to missing indexes or poor schema design.",
                "* High Memory Usage: Application consumes excessive memory, leading to crashes.",
                "* Long Load Times: Web pages or apps take too long to start or respond.",
                "* Poor Scalability: Performance drops significantly as user count increases."
            ]),
            ("4. Compatibility Issues", [
                "* Browser-Specific Bugs: Features work in one browser but fail in another (e.g., CSS issues in Internet Explorer).",
                "* Responsive Design Problems: Layouts break on devices with smaller screens or specific resolutions.",
                "* Outdated Libraries: Incompatibility caused by using deprecated libraries or APIs.",
                "* Device Dependency: Certain features require hardware not universally available (e.g., requiring NFC on a device)."
            ]),
            ("5. Functionality Bugs", [
                "* Logic Errors: Features behave incorrectly (e.g., a 'Save' button doesn't save data).",
                "* Unexpected Crashes: Application crashes during specific actions or edge cases.",
                "* Input Validation Failures: Accepting invalid data or rejecting valid data unnecessarily.",
                "* Feature Misalignment: Implementation does not match the requirements or user expectations."
            ])
        ]

        for section, points in feedback_points:
            section_label = Label(self.bullets_frame, text=section, font=("Roboto Condensed", 22, "bold"), bg="#D3D3D3", anchor="w", justify="left", fg="black")
            section_label.pack(anchor="w", padx=10)
            
            for point in points:
                bullet_label = Label(self.bullets_frame, text=point, font=("Raleway", 17), bg="#D3D3D3", wraplength=900, justify="left", anchor="w", fg="black")
                bullet_label.pack(anchor="w", padx=10)

        # Create a canvas in the center column to draw the black line (divider)
        self.canvas = tk.Canvas(self.root, bg="#D3D3D3", width=5)  # Canvas with width for divider line
        self.canvas.grid(row=0, column=1, sticky="ns")  # Positioned in the center column, spans vertically
        self.canvas.create_line(0, 0, 0, self.root.winfo_height(), fill="black", width=5)  # Black vertical line

        # Right-side frame for video feed and buttons
        self.right_frame = tk.Frame(self.root, bg="#D3D3D3")
        self.right_frame.grid(row=0, column=2, sticky="nsew", padx=20, pady=20)

        # Configure the right frame grid layout
        self.right_frame.grid_rowconfigure(0, weight=3)  # Video feed
        self.right_frame.grid_rowconfigure(1, weight=1)  # Gesture label
        self.right_frame.grid_rowconfigure(2, weight=1)  # Buttons
        self.right_frame.grid_rowconfigure(3, weight=1)  # Space feedback label
        self.right_frame.grid_columnconfigure(0, weight=1)

        # Video label for the camera feed
        self.video_label = Label(self.right_frame, bg="#D3D3D3")
        self.video_label.grid(row=0, column=0, padx=20, pady=3, sticky="nsew")

        # Real-time gesture label
        self.gesture_label = Label(self.right_frame, text="FEEDBACK : None", font=("Arial", 18), bg="#D3D3D3", fg="black")
        self.gesture_label.grid(row=1, column=0, padx=20, pady=3)

        # Start and Quit buttons
        self.button_frame = tk.Frame(self.right_frame, bg="#D3D3D3")
        self.button_frame.grid(row=2, column=0, pady=3)

        self.start_button = Button(self.button_frame, text="Start Detection", command=self.start_detection, font=("Arial", 15), bg="#4CAF50", fg="black", height=2, width=20)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.quit_button = Button(self.button_frame, text="Quit", command=self.quit_app, font=("Arial", 15), bg="#FF6347", fg="black", height=2, width=20)
        self.quit_button.pack(side=tk.LEFT, padx=10)

        # Spacebar feedback label
        self.space_gesture_label = Label(self.right_frame, text="(Click Space Bar to submit): None", font=("Arial", 18), bg="#000000", fg="white")
        self.space_gesture_label.grid(row=3, column=0, padx=20, pady=5)

        # Variables and threading
        self.running = False
        self.space_pressed = False
        self.current_gesture = "No gesture detected"
        self.cap = None
        self.thread = None

        # Bind space bar event
        self.root.bind('<space>', self.spacebar_pressed)

    def start_detection(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.gesture_label.config(text="Error: Unable to access camera!")
                self.running = False
                return
            # Start video detection in a separate thread
            self.thread = threading.Thread(target=self.detect_gesture, daemon=True)
            self.thread.start()

    def spacebar_pressed(self, event):
        self.space_pressed = True

    def detect_gesture(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        mp_draw = mp.solutions.drawing_utils

        confidence_threshold = 0.7

        while self.running:
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)  # Flip frame horizontally
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hand landmarks
            result = hands.process(img_rgb)

            last_gesture = "No gesture detected"

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])

                    # Flatten landmarks for model input
                    landmarks = np.array(landmarks).flatten().reshape(1, -1)
                    prediction = model.predict(landmarks, verbose=0)
                    class_id = np.argmax(prediction)
                    confidence = prediction[0][class_id]

                    if confidence > confidence_threshold and class_id in gesture_map:
                        last_gesture = gesture_map[class_id]
                    else:
                        last_gesture = "Gesture not valid"

                    # Draw hand landmarks
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Update real-time gesture label
            self.update_gesture_label(last_gesture)

            # Update spacebar label
            if self.space_pressed:
                self.update_space_gesture_label(last_gesture)
                self.space_pressed = False

            # Resize frame for larger display
            resized_frame = cv2.resize(frame, (440, 280))  # Adjust dimensions for larger display
            img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        hands.close()
        self.cap.release()

    def update_gesture_label(self, gesture):
        self.gesture_label.config(text=f"FEEDBACK : {gesture}")
    
    def update_space_gesture_label(self, gesture):
        self.space_gesture_label.config(text=f"(Submitted Feedback): {gesture}")

    def quit_app(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()

    # Optimize for MacBook screen size
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}")

    app = HandGestureApp(root)
    root.mainloop()
