import numpy as np
import cv2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# Load DenseNet121 model (without top classification layer)
base_model = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
# Now base_model outputs a 1024-d feature vector

def extract_densenet_features(image_paths):
    """
    image_paths: list of image paths
    returns: numpy array of feature vectors
    """
    features = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # Convert to 3-channel RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Resize to 224x224 (DenseNet input)
        img_rgb = cv2.resize(img_rgb, (224, 224))

        # Convert to array and expand dims
        img_arr = img_to_array(img_rgb)
        img_arr = np.expand_dims(img_arr, axis=0)

        # Preprocess for DenseNet
        img_arr = preprocess_input(img_arr)

        # Extract features
        feat = base_model.predict(img_arr, verbose=0)
        features.append(feat.flatten())

    return np.array(features)
