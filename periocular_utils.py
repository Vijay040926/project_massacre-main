# periocular_utils.py

import cv2
import mediapipe as mp
import numpy as np
import os
import json

# --- Configuration ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# MediaPipe 478 Landmark Indices for Periocular Region
# We select a wide area including eyes, eyebrows, and the bridge of the nose.
# --- THIS IS THE NEW, CORRECTED CODE ---
# Combine all the edge tuples from MediaPipe's constants
all_edges = (
    mp_face_mesh.FACEMESH_LEFT_EYE |
    mp_face_mesh.FACEMESH_LEFT_EYEBROW |
    mp_face_mesh.FACEMESH_RIGHT_EYE |
    mp_face_mesh.FACEMESH_RIGHT_EYEBROW
)

# Unroll the tuples (edges) into a single list of unique integer indices
unrolled_indices = {index for edge in all_edges for index in edge}

# Add the custom indices for the nose bridge and cheeks
unrolled_indices.update([6, 168, 195, 197])

# Finally, create the sorted list
PERIOCULAR_LANDMARK_INDICES = sorted(list(unrolled_indices))

# Landmark indices for occlusion detection
NOSE_TIP_INDEX = 4
UPPER_LIP_INDEX = 13
LOWER_LIP_INDEX = 14

# --- Core Functions ---

def get_face_landmarks(image):
    """
    Detects facial landmarks from a single image.
    Args:
        image (np.array): The input image in BGR format.
    Returns:
        A MediaPipe landmarks object if a face is found, otherwise None.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    return None

def is_occluded(landmarks, image_shape):
    """
    Detects if a face is likely occluded by a mask using a more robust heuristic.
    Checks visibility of key lower-face points.
    """
    if not landmarks:
        return False

    try:
        # Get the visibility of key points from MediaPipe
        nose_tip_vis = landmarks.landmark[NOSE_TIP_INDEX].visibility
        upper_lip_vis = landmarks.landmark[UPPER_LIP_INDEX].visibility
        lower_lip_vis = landmarks.landmark[LOWER_LIP_INDEX].visibility

        # If the nose and lips have low visibility, it's a very strong sign of a mask.
        # We check if the average visibility of these points is below a threshold.
        avg_lower_face_visibility = (nose_tip_vis + upper_lip_vis + lower_lip_vis) / 3.0

        if avg_lower_face_visibility < 0.6: # This threshold is more robust
            return True

    except IndexError:
        return False # Should not happen if landmarks are valid

    return False


def get_periocular_landmarks(landmarks, image_shape):
    """
    Extracts the (x, y) coordinates of the defined periocular landmarks.
    Args:
        landmarks: MediaPipe landmarks object.
        image_shape (tuple): The (height, width) of the image.
    Returns:
        np.array: An array of shape (N, 2) with periocular landmark coordinates.
    """
    h, w = image_shape[:2]
    coords = []
    for idx in PERIOCULAR_LANDMARK_INDICES:
        lm = landmarks.landmark[idx]
        coords.append([lm.x * w, lm.y * h])
    return np.array(coords, dtype=np.float32)

def vectorize_landmarks(landmark_coords):
    """
    Converts periocular landmark coordinates into a fixed-size, normalized vector.
    Normalization makes the vector robust to face scale and position.
    Args:
        landmark_coords (np.array): An array of shape (N, 2).
    Returns:
        np.array: A 1D normalized feature vector.
    """
    # 1. Center the landmarks around the mean (translation invariance)
    mean_point = landmark_coords.mean(axis=0)
    centered_coords = landmark_coords - mean_point
    
    # 2. Scale by the standard deviation (scale invariance)
    # Using L2 norm of all points from the center as the scaling factor
    scale = np.linalg.norm(centered_coords)
    if scale == 0: return None # Avoid division by zero
    
    normalized_coords = centered_coords / scale
    
    # 3. Flatten to create the final feature vector
    return normalized_coords.flatten()

def extract_and_save_periocular(image, person_id, base_filename, save_dir='dataset'):
    """
    Full pipeline to extract periocular data from an image and save it.
    Args:
        image (np.array): Cropped face image.
        person_id (str): The name or ID of the person.
        base_filename (str): The base name for the saved files (e.g., "00001").
        save_dir (str): The root dataset directory.
    Returns:
        dict or None: Metadata about the saved files or None on failure.
    """
    landmarks = get_face_landmarks(image)
    if not landmarks:
        return None

    # Define the save path for periocular data
    periocular_path = os.path.join(save_dir, person_id, 'periocular')
    os.makedirs(periocular_path, exist_ok=True)

    # Extract coordinates and create vector
    peri_coords = get_periocular_landmarks(landmarks, image.shape)
    peri_vector = vectorize_landmarks(peri_coords)
    
    if peri_vector is None:
        return None

    # Save the vector and landmark coordinates
    vector_filepath = os.path.join(periocular_path, f"{base_filename}.npy")
    coords_filepath = os.path.join(periocular_path, f"{base_filename}.json")

    np.save(vector_filepath, peri_vector)
    with open(coords_filepath, 'w') as f:
        json.dump(peri_coords.tolist(), f)
    
    occlusion_status = is_occluded(landmarks, image.shape)

    return {
        "person_id": person_id,
        "vector_file": vector_filepath,
        "coords_file": coords_filepath,
        "vector_shape": peri_vector.shape,
        "was_occluded": occlusion_status
    }