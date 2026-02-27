import os
import numpy as np
import glob
import time
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ttest_rel, wilcoxon

from densenet_extractor import extract_densenet_features
from sift_glcm_extractor import extract_sift_glcm_features

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# -------------------------------
# Image Perturbations (Robustness Testing)
# -------------------------------
import cv2

def add_noise(img, std=25):
    noise = np.random.normal(0, std, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_blur(img, k=5):
    return cv2.GaussianBlur(img, (k, k), 0)

def change_contrast(img, alpha=0.7):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def reduce_resolution(img, scale=0.5):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w*scale), int(h*scale)))
    return cv2.resize(small, (w, h))

def create_corrupted_dataset(image_paths, transform_fn, suffix):
    new_paths = []

    temp_dir = f"temp_{suffix}"
    os.makedirs(temp_dir, exist_ok=True)

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_corrupt = transform_fn(img)

        new_path = os.path.join(temp_dir, os.path.basename(path))
        cv2.imwrite(new_path, img_corrupt)
        new_paths.append(new_path)

    return new_paths


def print_metrics(name, y_true, y_pred):#Confusion matrix helper
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn)

    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Recall / Sensitivity: {recall:.4f}")
    print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")

def project_2d(X, method="pca", random_state=42):
    if method == "pca":
        return PCA(n_components=2).fit_transform(X)

    if method == "tsne":
        return TSNE(n_components=2, perplexity=30, random_state=random_state).fit_transform(X)

    if method == "umap":
        return umap.UMAP(n_components=2, random_state=random_state).fit_transform(X)

def plot_projection(X_2d, y, title):
    plt.figure(figsize=(6, 5))
    plt.scatter(
        X_2d[y == 0, 0], X_2d[y == 0, 1],
        c='blue', label='Class 0 (Non-fracture)', alpha=0.6
    )
    plt.scatter(
        X_2d[y == 1, 0], X_2d[y == 1, 1],
        c='red', label='Class 1 (Fracture)', alpha=0.6
    )
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------
# Load labels
# -------------------------------
def load_image_paths_and_labels(images_dir, labels_dir):
    image_paths, labels = [], []
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))

    for img_path in image_files:
        filename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(filename)[0]
        label_file = os.path.join(labels_dir, f"{name_without_ext}.txt")

        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    class_id = int(lines[0].split()[0])
                    label = 1 if class_id == 9 else 0 if class_id == 8 else None
                    if label is not None:
                        labels.append(label)
                        image_paths.append(img_path)

    return image_paths, labels


# -------------------------------
# Directories
# -------------------------------
train_images_dir = os.path.join("Human Bone Fractures Multi-modal Image Dataset (HBFMID)", "Bone Fractures Detection",
                                "train", "images")
train_labels_dir = os.path.join("Human Bone Fractures Multi-modal Image Dataset (HBFMID)", "Bone Fractures Detection",
                                "train", "labels")
valid_images_dir = os.path.join("Human Bone Fractures Multi-modal Image Dataset (HBFMID)", "Bone Fractures Detection",
                                "valid", "images")
valid_labels_dir = os.path.join("Human Bone Fractures Multi-modal Image Dataset (HBFMID)", "Bone Fractures Detection",
                                "valid", "labels")

# Load
train_image_paths, train_labels = load_image_paths_and_labels(train_images_dir, train_labels_dir)
valid_image_paths, valid_labels = load_image_paths_and_labels(valid_images_dir, valid_labels_dir)

# Combine train + valid
image_paths = train_image_paths + valid_image_paths
labels = train_labels + valid_labels

# Split train/val into new train/test
X_train, X_test, y_train, y_test = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# -------------------------------
# Feature extraction
# -------------------------------
print("Extracting DenseNet features...")
X_train_densenet = extract_densenet_features([str(path) for path in X_train])
X_test_densenet = extract_densenet_features([str(path) for path in X_test])

print("Extracting SIFT+GLCM features...")
X_train_siftglcm = extract_sift_glcm_features([str(path) for path in X_train])
X_test_siftglcm = extract_sift_glcm_features([str(path) for path in X_test])

# Combine train + test for visualization
X_all_densenet = np.vstack([X_train_densenet, X_test_densenet])
X_all_siftglcm = np.vstack([X_train_siftglcm, X_test_siftglcm])
y_all = np.array(y_train + y_test)

# -------------------------------
# DenseNet Feature Projections (Global Fracture Shape)
# -------------------------------
for method in ["pca", "tsne", "umap"]:
    X_2d = project_2d(X_all_densenet, method)
    plot_projection(
        X_2d, y_all,
        title=f"DenseNet Features ({method.upper()}) – Global Fracture Shape"
    )

# -------------------------------
# SIFT + GLCM Feature Projections (Texture & Edges)
# -------------------------------
for method in ["pca", "tsne", "umap"]:
    X_2d = project_2d(X_all_siftglcm, method)
    plot_projection(
        X_2d, y_all,
        title=f"SIFT + GLCM Features ({method.upper()}) – Texture & Edge Patterns"
    )

# -------------------------------
# 5-Fold Cross-Validation
# -------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_cnn = []
acc_sift = []
acc_stacked = []

for fold, (train_idx, test_idx) in enumerate(skf.split(image_paths, labels)):
    print(f"\nRunning Fold {fold + 1}/5")

    # Split paths and labels
    X_tr = [image_paths[i] for i in train_idx]
    X_te = [image_paths[i] for i in test_idx]
    y_tr = [labels[i] for i in train_idx]
    y_te = [labels[i] for i in test_idx]

    # Extract features
    X_tr_dn = extract_densenet_features(X_tr)
    X_te_dn = extract_densenet_features(X_te)

    X_tr_sg = extract_sift_glcm_features(X_tr)
    X_te_sg = extract_sift_glcm_features(X_te)

    # ---------------- CNN → SVM ----------------
    svm = SVC(kernel='rbf', C=1.0, probability=True)
    svm.fit(X_tr_dn, y_tr)
    preds_cnn = svm.predict(X_te_dn)
    acc_cnn.append(accuracy_score(y_te, preds_cnn))

    # ---------------- SIFT → FNN ----------------
    y_tr_cat = to_categorical(y_tr, num_classes=2)

    fnn_cv = Sequential([
        Input(shape=(X_tr_sg.shape[1],)),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dense(2, activation='softmax')
    ])
    fnn_cv.compile(optimizer='adam', loss='categorical_crossentropy')

    fnn_cv.fit(X_tr_sg, y_tr_cat, epochs=5, batch_size=2, verbose=0)
    preds_sift = np.argmax(fnn_cv.predict(X_te_sg, verbose=0), axis=1)
    acc_sift.append(accuracy_score(y_te, preds_sift))

    # ---------------- Stacked ----------------
    svm_probs_tr = svm.predict_proba(X_tr_dn)
    svm_probs_te = svm.predict_proba(X_te_dn)

    fnn_probs_tr = fnn_cv.predict(X_tr_sg, verbose=0)
    fnn_probs_te = fnn_cv.predict(X_te_sg, verbose=0)

    stacked_tr = np.hstack([svm_probs_tr, fnn_probs_tr])
    stacked_te = np.hstack([svm_probs_te, fnn_probs_te])

    meta = RandomForestClassifier(n_estimators=50, random_state=42)
    meta.fit(stacked_tr, y_tr)
    preds_stack = meta.predict(stacked_te)

    acc_stacked.append(accuracy_score(y_te, preds_stack))

# -------------------------------
# Statistical Significance Testing
# -------------------------------
def mean_ci(scores):
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    ci = 1.96 * std / np.sqrt(len(scores))
    return mean, ci

cnn_mean, cnn_ci = mean_ci(acc_cnn)
sift_mean, sift_ci = mean_ci(acc_sift)
stack_mean, stack_ci = mean_ci(acc_stacked)

# Paired tests
t_cnn_vs_stack = ttest_rel(acc_cnn, acc_stacked)
t_sift_vs_stack = ttest_rel(acc_sift, acc_stacked)

w_cnn_vs_stack = wilcoxon(acc_cnn, acc_stacked)
w_sift_vs_stack = wilcoxon(acc_sift, acc_stacked)

# -------------------------------
# DenseNet → SVM
# -------------------------------
svm_clf = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
svm_clf.fit(X_train_densenet, y_train)

svm_preds_train = svm_clf.predict_proba(X_train_densenet)
svm_preds_test = svm_clf.predict_proba(X_test_densenet)

svm_test_preds = svm_clf.predict(X_test_densenet)

svm_acc = accuracy_score(y_test, svm_test_preds)
svm_recall = recall_score(y_test, svm_test_preds, pos_label=1)
# -------------------------------
# SIFT+GLCM → FNN (5 epochs default)
# -------------------------------
y_train_cat = to_categorical(y_train, num_classes=2)

fnn = Sequential([
    Input(shape=(X_train_siftglcm.shape[1],)),
    Dense(64, activation='relu', kernel_regularizer='l2'),
    Dense(2, activation='softmax')
])
fnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

fnn.fit(X_train_siftglcm, y_train_cat, epochs=5, batch_size=2, verbose=1)

fnn_preds_train = fnn.predict(X_train_siftglcm)
fnn_preds_test = fnn.predict(X_test_siftglcm)

fnn_test_preds = np.argmax(fnn_preds_test, axis=1)

fnn_acc = accuracy_score(y_test, fnn_test_preds)
fnn_recall = recall_score(y_test, fnn_test_preds, pos_label=1)
# -------------------------------
# Stacking
# -------------------------------
stacked_train = np.hstack([svm_preds_train, fnn_preds_train])
stacked_test = np.hstack([svm_preds_test, fnn_preds_test])

meta_rf = RandomForestClassifier(n_estimators=50, random_state=42)
meta_rf.fit(stacked_train, y_train)
final_preds_rf = meta_rf.predict(stacked_test)

rf_acc = accuracy_score(y_test, final_preds_rf)
rf_recall = recall_score(y_test, final_preds_rf, pos_label=1)

meta_lr = LogisticRegression(max_iter=500, random_state=42)
meta_lr.fit(stacked_train, y_train)
final_preds_lr = meta_lr.predict(stacked_test)
lr_acc = accuracy_score(y_test, final_preds_lr)
lr_recall = recall_score(y_test, final_preds_lr, pos_label=1)


# -------------------------------
# Robustness Evaluation
# -------------------------------
def evaluate_pipeline(test_paths, y_true):

    # Feature extraction
    X_dn = extract_densenet_features(test_paths)
    X_sg = extract_sift_glcm_features(test_paths)

    # CNN → SVM
    cnn_preds = svm_clf.predict(X_dn)
    acc_cnn = accuracy_score(y_true, cnn_preds)

    # SIFT → FNN
    sg_probs = fnn.predict(X_sg, verbose=0)
    sg_preds = np.argmax(sg_probs, axis=1)
    acc_sift = accuracy_score(y_true, sg_preds)

    # Stacked
    dn_probs = svm_clf.predict_proba(X_dn)
    stacked = np.hstack([dn_probs, sg_probs])
    stack_preds = meta_rf.predict(stacked)
    acc_stack = accuracy_score(y_true, stack_preds)

    return acc_cnn, acc_sift, acc_stack

# -------------------------------
# 5 epochs vs 10 epochs FNN
# -------------------------------
results_epochs = {}

for epochs in [5, 10]:
    fnn_tmp = Sequential([
        Input(shape=(X_train_siftglcm.shape[1],)),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dense(2, activation='softmax')
    ])
    fnn_tmp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    start = time.time()
    fnn_tmp.fit(X_train_siftglcm, y_train_cat, epochs=epochs, batch_size=2, verbose=0)
    end = time.time()

    preds_tmp = fnn_tmp.predict(X_test_siftglcm, verbose=0)
    acc_tmp = accuracy_score(y_test, np.argmax(preds_tmp, axis=1))
    results_epochs[epochs] = {"acc": acc_tmp, "time": end - start}

# -------------------------------
# Plot 1: Final classifier RF vs LR
# -------------------------------
plt.figure(figsize=(6, 4))
plt.bar(["RandomForest", "LogReg"], [rf_acc, lr_acc], color=['skyblue', 'salmon'])
plt.title("Final Classifier: RF vs LR")
plt.ylabel("Accuracy")
plt.show()

# -------------------------------
# Plot 2: Individual vs Stacked
# -------------------------------
plt.figure(figsize=(6, 4))
plt.bar(["SIFT→FNN", "CNN→SVM", "Stacked"], [fnn_acc, svm_acc, rf_acc], color=['orange', 'green', 'blue'])
plt.title("SIFT→FNN vs CNN→SVM vs Stacked")
plt.ylabel("Accuracy")
plt.show()

# -------------------------------
# Plot 3: 5 vs 10 epochs (acc & time)
# -------------------------------
labels = ["5 epochs", "10 epochs"]
accs = [results_epochs[5]["acc"], results_epochs[10]["acc"]]
times = [results_epochs[5]["time"], results_epochs[10]["time"]]

x = np.arange(len(labels))
width = 0.35

fig, ax1 = plt.subplots(figsize=(7, 4))
bar1 = ax1.bar(x - width / 2, accs, width, label='Accuracy')
bar2 = ax1.bar(x + width / 2, times, width, label='Time (s)')

ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_title("5 vs 10 epochs: Accuracy & Time")
ax1.legend()
plt.show()

print(
    f"\nAccuracy | Recall (Sensitivity)\n"
    f"SVM: {svm_acc:.4f} | {svm_recall:.4f}\n"
    f"FNN: {fnn_acc:.4f} | {fnn_recall:.4f}\n"
    f"RF-stacked: {rf_acc:.4f} | {rf_recall:.4f}\n"
    f"LR-stacked: {lr_acc:.4f} | {lr_recall:.4f}"
)
print("Epoch comparison:", results_epochs)
print_metrics("SVM (DenseNet)", y_test, svm_test_preds)
print_metrics("FNN (SIFT+GLCM)", y_test, fnn_test_preds)
print_metrics("Stacked RF", y_test, final_preds_rf)
print_metrics("Stacked LR", y_test, final_preds_lr)

print("\n===== 5-Fold Cross-Validation Results =====")
print(f"CNN (DenseNet): {cnn_mean:.4f} ± {cnn_ci:.4f}")
print(f"SIFT+GLCM:     {sift_mean:.4f} ± {sift_ci:.4f}")
print(f"Stacked:       {stack_mean:.4f} ± {stack_ci:.4f}")

print("\n===== Statistical Tests =====")
print("Paired t-test:")
print(f"CNN vs Stacked:  p = {t_cnn_vs_stack.pvalue:.4e}")
print(f"SIFT vs Stacked: p = {t_sift_vs_stack.pvalue:.4e}")

print("\nWilcoxon test:")
print(f"CNN vs Stacked:  p = {w_cnn_vs_stack.pvalue:.4e}")
print(f"SIFT vs Stacked: p = {w_sift_vs_stack.pvalue:.4e}")

print("\n===== ROBUSTNESS TESTING =====")

perturbations = {
    "Noise": lambda img: add_noise(img, 25),
    "Blur": lambda img: add_blur(img, 5),
    "LowContrast": lambda img: change_contrast(img, 0.6),
    "LowResolution": lambda img: reduce_resolution(img, 0.5)
}

robust_results = {}

# Original baseline
robust_results["Original"] = evaluate_pipeline(X_test, y_test)

for name, fn in perturbations.items():
    print(f"Applying {name}...")
    corrupted_paths = create_corrupted_dataset(X_test, fn, name)

    robust_results[name] = evaluate_pipeline(corrupted_paths, y_test)

print("\n===== Robustness Results =====")
print("Condition | CNN | SIFT | Stacked")

for k, v in robust_results.items():
    print(f"{k:12s} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f}")


conditions = list(robust_results.keys())
cnn_vals = [robust_results[c][0] for c in conditions]
sift_vals = [robust_results[c][1] for c in conditions]
stack_vals = [robust_results[c][2] for c in conditions]

plt.figure(figsize=(8,5))
plt.plot(conditions, cnn_vals, marker='o', label="CNN→SVM")
plt.plot(conditions, sift_vals, marker='o', label="SIFT→FNN")
plt.plot(conditions, stack_vals, marker='o', label="Stacked")

plt.title("Model Robustness Under Image Perturbations")
plt.ylabel("Accuracy")
plt.xlabel("Perturbation Type")
plt.legend()
plt.grid(True)
plt.show()
