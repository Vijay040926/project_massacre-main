import numpy as np
import imutils
import time
import cv2
import csv
import os
import os
from periocular_utils import extract_and_save_periocular

# --- Configuration ---
# File path for the Haar Cascade classifier
cascade = 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade)
DATASET_PATH = 'dataset'

# We still aim for 50 distinct CAPTURE MOMENTS, but each moment now produces 3 images.
NUM_CAPTURE_MOMENTS = 50 
TOTAL_IMAGES_SAVED = NUM_CAPTURE_MOMENTS * 3

FACE_PADDING = 50 # Pixels of margin to add around the detected face for context
GAMMA_VALUE = 0.5 # Value for low-light correction simulation (less than 1 makes image brighter)

# Define the different poses and the number of samples for each
# Note: samples refers to the number of *capture moments* needed for this pose.
poses = [
    {"name": "Frontal Face", "samples": 10, "instructions": "Look straight at the camera."},
    {"name": "Slight Left Turn", "samples": 10, "instructions": "Turn your head slightly to the left."},
    {"name": "Slight Right Turn", "samples": 10, "instructions": "Turn your head slightly to the right."},
    {"name": "Slight Upwards Tilt", "samples": 10, "instructions": "Tilt your head slightly up."},
    {"name": "Slight Downwards Tilt", "samples": 10, "instructions": "Tilt your head slightly down."},
]

# --- Helper Functions ---

def adjust_gamma(image, gamma=1.0):
    """
    Applies gamma correction to an image. Used to simulate or correct low-light conditions.
    """
    # build a lookup table mapping the input pixels to the output pixel values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def draw_feedback(frame, text, color=(0, 255, 0), position_y_offset=10):
    """Draws feedback text on the frame with a background for visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    text_x = 10
    text_y = frame.shape[0] - position_y_offset
    
    # Draw a background rectangle
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    
    # Put the text
    cv2.putText(frame, text, (text_x, text_y), font, 0.7, color, 2)

def draw_perfect_position_guide(frame, rects, color=(0, 255, 255)):
    """
    Draws a guide rectangle and checks if the detected face is centered within it.
    Returns True if the face is centered and of a decent size, False otherwise.
    """
    H, W = frame.shape[:2]
    
    # Define an ideal centered region (e.g., 40% of frame width, 50% of frame height)
    ideal_w = int(W * 0.4)
    ideal_h = int(H * 0.5)
    ideal_x = int((W - ideal_w) / 2)
    ideal_y = int((H - ideal_h) / 2)
    
    # Draw the guide rectangle
    cv2.rectangle(frame, (ideal_x, ideal_y), (ideal_x + ideal_w, ideal_y + ideal_h), color, 2)
    cv2.putText(frame, "Ideal Face Position", (ideal_x, ideal_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    is_centered = False
    
    if rects:
        (x, y, w, h) = rects[0] # Assume the largest face is the target
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Check if the face center is within the ideal region boundaries
        is_centered = (face_center_x >= ideal_x and face_center_x <= ideal_x + ideal_w and
                       face_center_y >= ideal_y and face_center_y <= ideal_y + ideal_h)
        
        is_large_enough = h > ideal_h * 0.3 # Face must be a decent size

        if is_centered and is_large_enough:
            draw_feedback(frame, "PERFECT POSITION - AUGMENTING DATA!", (0, 255, 0), 40)
            return True
        elif not is_large_enough:
             draw_feedback(frame, "MOVE CLOSER!", (0, 0, 255), 40)
        else:
            # Face is detected but not centered
            draw_feedback(frame, "MOVE TO THE YELLOW BOX!", (0, 0, 255), 40)
            
    return False


# --- Data Input and Setup ---
while True:
    try:
        Name = str(input("Enter your Name: ").strip())
        if not Name:
            print("Name cannot be empty. Please try again.")
            continue
        Roll_Number = int(input("Enter your Roll_Number: "))
        break
    except ValueError:
        print("Invalid Roll Number. Please enter an integer.")

sub_data = Name
path = os.path.join(DATASET_PATH, sub_data)

if not os.path.isdir(path):
    os.makedirs(path, exist_ok=True)
    print(f"Created directory: {path}")

# Write student info to CSV
info = [str(Name), str(Roll_Number)]
with open('student.csv', 'a', newline='') as csvFile: 
    write = csv.writer(csvFile)
    write.writerow(info)
print("Student information saved to student.csv.")

print("Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(1.0) # Shorter delay

# --- Guided Data Collection Loop ---
global_capture_count = 0 # Tracks the number of successful capture moments (out of NUM_CAPTURE_MOMENTS)
global_image_count = 0 # Tracks the total number of images saved (out of TOTAL_IMAGES_SAVED)

for pose_data in poses:
    pose_name = pose_data["name"]
    pose_samples = pose_data["samples"]
    pose_instructions = pose_data["instructions"]
    
    if global_capture_count >= NUM_CAPTURE_MOMENTS:
        break
        
    print(f"\n--- Starting: {pose_name} ({pose_samples} capture moments) ---")
    
    current_pose_count = 0
    
    print(f"INSTRUCTION: {pose_instructions}. Look at the screen and press ENTER to start capturing for this pose.")
    input()

    while current_pose_count < pose_samples:
        _, frame = cam.read()
        frame = imutils.resize(frame, width=640)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        rects = sorted(rects, key=lambda x: x[2] * x[3], reverse=True)

        is_centered = draw_perfect_position_guide(frame, rects)
        
        # Display current status and instructions
        status_text = f"Pose: {pose_name} | Captured: {global_capture_count}/{NUM_CAPTURE_MOMENTS} Moments | Total Files: {global_image_count}/{TOTAL_IMAGES_SAVED}"
        draw_feedback(frame, status_text, color=(255, 255, 0), position_y_offset=10)
        draw_feedback(frame, f"TASK: {pose_instructions}", (255, 0, 255), position_y_offset=70)


        # CAPTURE & AUGMENTATION LOGIC:
        if rects and is_centered:
            (x, y, w, h) = rects[0]
            
            # 1. Calculate padded crop area
            H, W = frame.shape[:2]
            x1 = max(0, x - FACE_PADDING)
            y1 = max(0, y - FACE_PADDING)
            x2 = min(W, x + w + FACE_PADDING)
            y2 = min(H, y + h + FACE_PADDING)
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                print("Warning: Face crop resulted in zero size, skipping capture.")
                time.sleep(0.1)
                continue

            # --- 2. AUGMENTATION AND SAVING ---
            base_filename = str(global_capture_count).zfill(5)
            
            # a) Normal Image (Base)
            cv2.imwrite(os.path.join(path, f"{base_filename}_normal.png"), face_crop)
            global_image_count += 1
            
            # b) Low-Light Robust Image (Gamma Correction)
            gamma_img = adjust_gamma(face_crop, gamma=GAMMA_VALUE)
            cv2.imwrite(os.path.join(path, f"{base_filename}_gamma.png"), gamma_img)
            global_image_count += 1

            # c) Pixelation Robust Image (Gaussian Blur)
            # Kernel size (e.g., 7, 7) determines blur intensity.
            blurred_img = cv2.GaussianBlur(face_crop, (7, 7), 0) 
            cv2.imwrite(os.path.join(path, f"{base_filename}_blur.png"), blurred_img)
            global_image_count += 1
            # --- END AUGMENTATION ---

            # --- 3. PERIOCULAR EXTRACTION ---
            # We extract periocular data from the original, clean face crop
            metadata = extract_and_save_periocular(face_crop, sub_data, base_filename, save_dir=DATASET_PATH)
            if metadata:
                print(f"  -> Saved periocular vector for {base_filename}.npy")
            else:
                print(f"  -> Warning: Could not extract periocular data for {base_filename}")
            # --- END PERIOCULAR EXTRACTION ---

            current_pose_count += 1
            global_capture_count += 1 # Only increment the capture moment count once
            
            # Visual confirmation (flash a green box briefly)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
            draw_feedback(frame, f"3 IMAGES SAVED! ({global_image_count} total)", (0, 255, 0), 40)
            
            time.sleep(0.5) # Longer pause to save 3 files and ensure user sees confirmation
        
        elif rects and not is_centered:
            # Draw the bounding box even if not centered, but in red
            (x, y, w, h) = rects[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            time.sleep(0.05)
        
        # Show the processed frame
        cv2.imshow("Dataset Creator (Robust Mode)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            global_capture_count = NUM_CAPTURE_MOMENTS 
            break
    
    if global_capture_count >= NUM_CAPTURE_MOMENTS:
        break

# --- Cleanup ---
cam.release()
cv2.destroyAllWindows()
print(f"\nData collection complete for {Name}.")
print(f"Total Capture Moments achieved: {global_capture_count}/{NUM_CAPTURE_MOMENTS}")
print(f"Total Images Saved (including augmentation): {global_image_count}/{TOTAL_IMAGES_SAVED}")

if global_capture_count < NUM_CAPTURE_MOMENTS:
    print("WARNING: Data collection was stopped early. The augmented dataset is incomplete.")
