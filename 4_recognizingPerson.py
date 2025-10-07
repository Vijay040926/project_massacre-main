import numpy as np
import imutils
import pickle
import time
import cv2
import csv
import os
from collections.abc import Iterable

# --- Configuration & Efficiency Tuning ---
TARGET_CONFIDENCE = 0.70             # 70% confidence required to mark attendance (Updated per your selection)
MAX_RUN_TIME_SECONDS = 30            # Stop after 10 seconds 
embeddingModel = "openface_nn4.small2.v1.t7"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.5                           # Detector confidence (kept for face detection)

# --- New Helper Function for Positional Guidance ---
def draw_guide_and_feedback(frame, rects):
    """
    Draws a guide box and provides textual feedback on face position.
    Returns True if the face is well-positioned, False otherwise.
    """
    H, W = frame.shape[:2]
    # Define an ideal centered region (e.g., 40% of frame width, 50% of frame height)
    ideal_w = int(W * 0.4)
    ideal_h = int(H * 0.5)
    ideal_x = int((W - ideal_w) / 2)
    ideal_y = int((H - ideal_h) / 2)
    
    GUIDE_COLOR = (255, 255, 0) # Yellow
    STATUS_POSITION = (10, H - 10)
    
    # Draw the static guide rectangle
    cv2.rectangle(frame, (ideal_x, ideal_y), (ideal_x + ideal_w, ideal_y + ideal_h), GUIDE_COLOR, 2)
    cv2.putText(frame, "IDEAL ZONE", (ideal_x, ideal_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GUIDE_COLOR, 2)
    
    is_well_positioned = False
    
    if rects:
        (x, y, w, h) = rects[0] # Focus on the largest detected face
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Check if the face center is within the ideal region boundaries
        is_centered = (face_center_x >= ideal_x and face_center_x <= ideal_x + ideal_w and
                       face_center_y >= ideal_y and face_center_y <= ideal_y + ideal_h)
        
        # Check if the face is large enough (e.g., at least 30% of the guide box height)
        is_large_enough = h > ideal_h * 0.3

        if is_centered and is_large_enough:
            guide_feedback = "POSITION: Perfect! Hold still."
            feedback_color = (0, 255, 0) # Green
            is_well_positioned = True
        elif not is_large_enough:
            guide_feedback = "POSITION: Move Closer (Face too small)"
            feedback_color = (0, 0, 255) # Red
        else:
            guide_feedback = "POSITION: Center your face in the YELLOW BOX!"
            feedback_color = (0, 0, 255) # Red
            
        # Draw dynamic feedback text
        cv2.putText(frame, guide_feedback, STATUS_POSITION, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2)
            
    return is_well_positioned

# --- Initialization: Load Models ---
print("[INFO] Loading Face Detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] Loading Face Recognizer and Embedder...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

try:
    recognizer = pickle.loads(open(recognizerFile, "rb").read())
    le = pickle.loads(open(labelEncFile, "rb").read())
except FileNotFoundError as e:
    print(f"[FATAL ERROR] Required model file not found: {e}. Please train the model first.")
    exit()

# --- Data Loading (Optimization) ---
# OPTIMIZATION: Load Roll Numbers and Names once efficiently into a dictionary
student_data = {}
try:
    with open('student.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            # Assumes CSV format: [Name, Roll_Number]
            if len(row) >= 2:
                # Use .strip() for robust matching against trained names
                student_data[row[0].strip()] = row[1].strip()
    print(f"[INFO] Loaded student data for {len(student_data)} unique individuals.")
except FileNotFoundError:
    print("[WARNING] student.csv not found. Roll Numbers will be listed as N/A.")

# --- Video Stream Setup ---
print(f"[INFO] Starting video stream for single-shot recognition (Max {MAX_RUN_TIME_SECONDS}s)...")
cam = cv2.VideoCapture(0)
time.sleep(1.0)

# --- Full-Screen Setup ---
# Create a named window and set it to full-screen property
cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# --- Main Recognition Loop (Single-Shot Optimization) ---
start_time = time.time()
attendance_marked = False
recognized_person = None # Will store (name, roll_number, proba) if successful

while (time.time() - start_time) < MAX_RUN_TIME_SECONDS and not attendance_marked:
    _, frame = cam.read()
    if frame is None:
        print("[ERROR] Failed to read frame from webcam.")
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    
    # Pre-process the frame for face detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()
    
    # Filter out low-confidence detections
    rects = []
    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            rects.append(box.astype("int"))
            
    # Sort and take the largest face for processing and guidance
    rects = sorted(rects, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]), reverse=True)
    
    # --- POSITIONAL GUIDANCE ---
    is_positioned_well = draw_guide_and_feedback(frame, rects)

    # Temporary display text for current frame (will be overwritten if a face is found)
    display_text = "STATUS: Searching..."
    display_color = (0, 0, 255) # Blue

    # Process only the largest face if one was found
    if rects:
        (startX, startY, endX, endY) = rects[0]

        face = frame[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        if fW < 20 or fH < 20:
            # Draw box for detected but too-small face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, "Face too small", (startX, startY - 10 if startY - 10 > 10 else startY + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
            # Skip recognition if size check fails
            continue 
            
        # Only attempt classification if the face is in the correct position for quality
        if is_positioned_well:

            # Compute facial embedding
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Classify
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name_raw = le.classes_[j]
            name = name_raw.strip()
            
            # --- Attendance Logic (70% confidence required) ---
            if proba >= TARGET_CONFIDENCE:
                # High-confidence detection achieved! Prepare to exit.
                attendance_marked = True
                roll_number = student_data.get(name, "N/A")
                recognized_person = (name, roll_number, proba)
            
            # --- Visual Feedback for Low Confidence ---
            display_text = f"{name}: {proba * 100:.2f}% (Target: {TARGET_CONFIDENCE * 100:.0f}%)"
            display_color = (0, 165, 255) # Orange

            # Draw box and text
            cv2.rectangle(frame, (startX, startY), (endX, endY), display_color, 2)
            cv2.putText(frame, display_text, (startX, startY - 10 if startY - 10 > 10 else startY + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, display_color, 2)
    
    # If attendance marked, break the outer loop immediately
    if attendance_marked:
        break

    # Update overall status text on the frame
    remaining_time = int(MAX_RUN_TIME_SECONDS - (time.time() - start_time))
    
    # Note: Positional feedback is drawn by draw_guide_and_feedback,
    # so we'll move the time remaining status slightly up.
    cv2.putText(frame, f"Time Left: {remaining_time}s", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # White status text
    
    cv2.imshow("Frame", frame)
    
    # FIX: Increase waitKey time to prevent system hanging
    key = cv2.waitKey(20) & 0xFF
    if key == 27: # ESC key allows manual stop if needed
        break

# --- Cleanup and Final Presentation (Single Accurate Answer) ---
cam.release()
# cv2.destroyAllWindows()

print("-" * 70)
if recognized_person:
    name, roll_number, proba = recognized_person
    print(f"[ATTENDANCE MARKED SUCCESSFULLY]")
    print(f"Name: {name}")
    print(f"Roll Number: {roll_number}")
    print(f"Status: PRESENT (Confidence: {proba * 100:.2f}%)")
else:
    print("[ATTENDANCE FAILED]")
    print(f"Status: NO HIGH-CONFIDENCE MATCH FOUND within {MAX_RUN_TIME_SECONDS} seconds.")
print("-" * 70)