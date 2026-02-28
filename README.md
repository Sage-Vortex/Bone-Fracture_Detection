# Bone-Fracture_Detection

# Hybrid Feature Learning and Robust Fracture Classification Pipeline

## Project Overview

This project implements a hybrid medical image classification framework for automated bone fracture detection using multimodal feature learning and ensemble modeling. The objective is to combine deep convolutional representations with handcrafted texture descriptors to capture complementary structural information present in radiographic imagery. Rather than relying solely on end-to-end deep learning, the pipeline integrates pretrained DenseNet feature extraction with classical computer vision descriptors (SIFT and GLCM) and evaluates how these heterogeneous representations contribute to classification accuracy, robustness, and statistical reliability.

The system is designed as a complete experimental pipeline: data loading, feature extraction, representation analysis, model training, cross-validation, statistical testing, robustness evaluation, and comparative visualization are unified into a single workflow. The framework targets binary fracture detection by learning discriminative patterns that separate fractured and non-fractured bone images while explicitly analyzing model stability under distribution shifts and image degradation.

---

## Data Preparation and Label Construction

Images are sourced from the Human Bone Fractures Multi-modal Image Dataset (HBFMID). Image paths and labels are automatically aligned by matching image filenames with annotation text files. Only selected class identifiers are retained, mapping fracture and non-fracture categories into a binary classification task.

Training and validation splits from the dataset are merged and re-partitioned into a stratified train–test split to ensure balanced class representation. This reorganization allows consistent evaluation across multiple experimental stages while preserving label distribution. Images are processed in grayscale to emphasize structural patterns rather than color information, reflecting the clinical nature of radiographic imagery.

---

## Feature Extraction Strategy

The pipeline employs two fundamentally different feature extraction paradigms.

**Deep Feature Representation (DenseNet121)**
A pretrained DenseNet121 network, truncated before the classification layer, serves as a high-level feature encoder. Each image is resized and converted into a three-channel representation before being passed through the network to obtain a global 1024-dimensional embedding. These features capture large-scale anatomical structure and global fracture morphology learned from ImageNet pretraining.

**Handcrafted Texture Representation (SIFT + GLCM)**
Classical descriptors are extracted to encode localized structural patterns:

* SIFT captures invariant keypoint gradients and edge geometry.
* GLCM statistics quantify texture relationships such as contrast, homogeneity, correlation, and energy.

The resulting feature vector combines local edge descriptors with statistical texture measures, providing information complementary to CNN embeddings.

The hybrid design reflects an explicit hypothesis: deep networks capture semantic structure, while handcrafted descriptors retain interpretable texture cues often relevant in medical imaging.

---

## Representation Analysis and Visualization

To analyze separability of learned representations, feature embeddings are projected into two dimensions using multiple manifold learning techniques:

* Principal Component Analysis (PCA)
* t-Distributed Stochastic Neighbor Embedding (t-SNE)
* Uniform Manifold Approximation and Projection (UMAP)

These projections provide qualitative insight into how fractures cluster within feature space and how global CNN features differ from texture-based descriptors. Visualization enables comparison of feature geometry rather than relying solely on classification metrics.

---

## Modeling Framework

Three classification paradigms are evaluated:

1. **DenseNet Features → Support Vector Machine (SVM)**
   A radial-basis SVM models nonlinear decision boundaries over deep feature embeddings.

2. **SIFT+GLCM Features → Feedforward Neural Network (FNN)**
   A shallow neural network learns discriminative combinations of handcrafted descriptors with L2 regularization to control overfitting.

3. **Stacked Ensemble Model**
   Probabilistic outputs from both models are concatenated and used as input to meta-learners:

   * Random Forest classifier
   * Logistic Regression classifier

This stacked architecture aims to combine complementary prediction signals and reduce individual model bias.

---

## Cross-Validation and Statistical Evaluation

Model performance is evaluated using stratified 5-fold cross-validation to estimate generalization stability. Accuracy scores from each fold are aggregated to compute mean performance and confidence intervals.

To verify whether improvements from stacking are statistically meaningful, paired hypothesis tests are conducted:

* Paired t-tests
* Wilcoxon signed-rank tests

These analyses assess whether ensemble gains exceed random variation across folds, introducing statistical rigor often absent from standard ML pipelines.

---

## Robustness and Perturbation Testing

A dedicated robustness module evaluates model sensitivity to realistic image degradations. Test images undergo controlled perturbations including:

* Gaussian noise injection
* Gaussian blur
* Contrast reduction
* Resolution downscaling

For each corruption type, the full inference pipeline is re-executed and accuracy is measured across individual and stacked models. This stage evaluates resilience to acquisition noise and domain shifts, approximating real-world clinical variability.

---

## Training Efficiency Analysis

The pipeline also investigates training trade-offs by comparing neural network performance under different epoch counts. Accuracy and runtime are jointly analyzed to quantify efficiency versus performance gains, highlighting diminishing returns in training duration.

---

## Experimental Outputs

The framework generates:

* Feature space projections for interpretability
* Accuracy comparisons between individual and ensemble models
* Cross-validation performance summaries
* Statistical significance results
* Robustness curves across perturbations
* Training efficiency comparisons

Together, these outputs provide both predictive evaluation and analytical understanding of model behavior.

---

## Contributions

* Hybrid integration of deep CNN embeddings and handcrafted vision descriptors
* Comparative analysis of global shape versus local texture representations
* Ensemble stacking across heterogeneous feature domains
* Statistical validation of model improvements
* Robustness benchmarking under controlled image perturbations
* Visualization-driven analysis of learned feature spaces

This project demonstrates that combining classical computer vision with modern deep representations can yield interpretable and resilient medical image classification systems while enabling systematic evaluation beyond simple accuracy metrics.
