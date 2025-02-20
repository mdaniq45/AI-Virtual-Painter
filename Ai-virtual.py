
import cv2
import mediapipe as mp
import numpy as np

# Initialize Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define Colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Blue, Green, Red, Yellow
color_index = 0
brush_size = 5

# Initialize canvas
canvas = None

# Initialize Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Create a blank canvas (same size as the webcam feed)
    if canvas is None:
        canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]  # Index Finger Tip
            x, y = int(index_finger_tip.x * frame_width), int(index_finger_tip.y * frame_height)
            
            # Draw on the canvas
            cv2.circle(canvas, (x, y), brush_size, colors[color_index], -1)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect if the user wants to change color (Middle Finger Up)
            middle_finger_tip = hand_landmarks.landmark[12]
            mx, my = int(middle_finger_tip.x * frame_width), int(middle_finger_tip.y * frame_height)

            if abs(x - mx) < 40 and abs(y - my) < 40:  # If index and middle finger are close
                color_index = (color_index + 1) % len(colors)  # Cycle through colors

            # Detect if the user wants to erase (Fist Gesture - All fingers close)
            thumb_tip = hand_landmarks.landmark[4]
            if abs(x - int(thumb_tip.x * frame_width)) < 40:
                cv2.circle(canvas, (x, y), brush_size * 2, (0, 0, 0), -1)  # Erase

            # Clear Screen (if palm is fully open)
            wrist = hand_landmarks.landmark[0]
            if abs(wrist.y - index_finger_tip.y) < 0.1:  # Hand is fully open
                canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Merge canvas with the frame
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display the output
    cv2.imshow("AI Virtual Painter", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
