import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


# Path to your images (update if needed)
base_dir = "Human Bone Fractures Multi-modal Image Dataset (HBFMID)/Bone Fractures Detection"
train_images_dir = os.path.join(base_dir, "train", "images")

# --- Image Preprocessing ---
def preprocess_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(img, (224, 224))
    return img  # ✅ stays uint8, good for SIFT+GLCM
# --- SIFT Feature Extraction ---
def extract_sift_features(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None:
        # Take mean of descriptors to get fixed-size vector
        sift_features = np.mean(descriptors, axis=0)
    else:
        # If no descriptors, fallback to zeros
        sift_features = np.zeros(128)
    return sift_features

# --- GLCM Feature Extraction ---
def extract_glcm_features(img):
    # GLCM expects uint8 levels 0-255, distances & angles can be tuned
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    ASM = graycoprops(glcm, 'ASM')[0, 0]
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation, ASM])

# --- Combined ---
def extract_sift_glcm_features(image_paths):
    features = []
    for image_path in image_paths:
        # Ensure the path is a string
        img_path = str(image_path)
        img = preprocess_image(img_path)

        # Extract SIFT features
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        sift_features = np.mean(des, axis=0) if des is not None else np.zeros(128)

        # Extract GLCM features
        glcm_features = extract_glcm_features(img)

        # Combine features
        combined_features = np.hstack([sift_features, glcm_features])
        features.append(combined_features)

    return np.array(features)


# --- Test ---
if __name__ == "__main__":
    filenames = os.listdir(train_images_dir)[:5]
    for fname in filenames:
        img_path = os.path.join(train_images_dir, fname)
        features = extract_sift_glcm_features([img_path])  #  pass as list!
        print(f"{fname} | Feature vector shape: {features.shape}")