import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os
import cv2  #Included for context, though not used in training the SVC

# --- Configuration ---
embeddingFile = "output/embeddings.pickle"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"

# --- Model Loading and Training ---

print("[INFO] Loading face embeddings...")
# This file contains the 'embeddings' (vectors) and 'names' (labels) generated
# from your image dataset.
try:
    data = pickle.loads(open(embeddingFile, "rb").read())
except FileNotFoundError:
    print(f"ERROR: {embeddingFile} not found. Please run the embedding generation step first.")
    exit()

# ML Engineering Strategy for Robustness:
# The robustness to low light and pixelation must be built into the embeddings
# themselves. This means that when generating the 'embeddings.pickle' file,
# the original images must have been pre-processed:
# 1. Low Light: Use CLAHE (Contrast Limited Adaptive Histogram Equalization) or
#    Gamma Correction before feeding the face to the OpenFace embedder.
# 2. Pixelation: Use aggressive data augmentation (blurring, downsampling,
#    and adding noise) to the training images to simulate pixelation.

print("[INFO] Encoding labels...")
# Converts text names (e.g., 'Joe', 'Jane') into numerical labels (e.g., 0, 1)
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(data["names"])

print("[INFO] Training model...")
# SVC is a great choice for classification on top of deep embeddings.
# We keep probability=True for reliability scoring (proba >= TARGET_CONFIDENCE).
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# --- Saving the Trained Model and Labels ---

print("[INFO] Saving recognizer...")
with open(recognizerFile, "wb") as f:
    f.write(pickle.dumps(recognizer))

print("[INFO] Saving label encoder...")
with open(labelEncFile, "wb") as f:
    f.write(pickle.dumps(labelEnc))

print("[SUCCESS] Training complete. Model and labels saved.")
