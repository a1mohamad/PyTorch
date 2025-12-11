# 📖 Chapter 14: Classifying Images with Deep Convolutional Neural Networks (CNNs)

This comprehensive README outlines the key theoretical concepts and practical implementations of **Chapter 14** of *Machine Learning with PyTorch and Scikit-Learn*. This chapter marks the major transition from general-purpose Multilayer Perceptrons (MLPs) to **Convolutional Neural Networks (CNNs)**—the specialized architecture essential for state-of-the-art image and sequential data processing.

---

## 🎯 Chapter Overview: The Shift to Spatial Feature Learning

The core objective of Chapter 14 is to master the layers and techniques that allow a network to learn hierarchical features directly from image pixels. We move from treating image pixels as a flat feature vector (as in MLPs) to preserving their 2D structure for more effective feature extraction.

| Focus Area | Core Concept | Significance |
| :--- | :--- | :--- |
| **Foundation** | Convolution and Pooling | Enables automatic feature extraction and spatial size reduction. |
| **Data Handling** | Color Tensors & Transforms | Efficiently loading and normalizing multi-channel (RGB) images. |
| **Loss Functions** | Numerical Stability | Using stable logit-based loss functions for binary and multiclass tasks. |
| **Application** | Deep CNN Architecture | Building and training powerful models for complex real-world image classification. |

---

## 🛠️ Section 1: The Core CNN Building Blocks (Notebooks: `deep_CNN.ipynb`, `MNIST_with_CNN.ipynb`)

This section introduces the essential layers that define the CNN architecture, moving from concept to implementation.

### 1. Convolutional Layer (`nn.Conv2d`)

* **Feature Extraction:** We understand how a small, learned matrix called a **kernel** (or filter) slides across the input image. 
* **Feature Maps:** This operation outputs **feature maps**, where each value is the result of the kernel detecting a specific pattern (like an edge or corner) in a region of the input.
* **Architecture Parameters:** We configure parameters like `in_channels` (e.g., 3 for RGB), `out_channels` (number of kernels), `kernel_size`, and `padding`.

### 2. Pooling Layer (`nn.MaxPool2d`)

* **Spatial Downsampling:** Pooling (usually Max Pooling) progressively reduces the height and width of the feature maps.  This provides two key benefits:
    1.  **Reduces Computation:** Fewer parameters need to be processed in later layers.
    2.  **Increased Robustness:** Makes the features less sensitive to the exact position of a feature in the input image.

### 3. Image Tensor Representation

* **Shape Convention:** We establish the standard PyTorch image format: **(Batch Size, Channels, Height, Width)** (N, C, H, W).
* **Grayscale vs. Color:** The initial notebook (`deep_CNN.ipynb`) verifies that grayscale images (like MNIST) have C=1, while color images (like CelebA) have C=3 (for RGB).

---

## ⚖️ Section 2: Numerically Stable Loss Functions (Notebook: `deep_CNN.ipynb`)

Before training, it is crucial to use the most stable loss functions, which operate on raw network outputs (**logits**) rather than normalized probabilities.

| Task | Preferred Loss (Stable) | Raw Input | Legacy Method (Less Stable) |
| :--- | :--- | :--- | :--- |
| **Binary** | **`nn.BCEWithLogitsLoss`** | Logits | `nn.BCELoss` (requires Sigmoid) |
| **Multiclass** | **`nn.CrossEntropyLoss`** | Logits | `nn.NLLLoss` (requires Log-Softmax) |

The notebook confirms that the preferred stable methods internally perform the necessary activation (Sigmoid or Softmax) and achieve the same result as the legacy two-step methods, but with fewer numerical pitfalls.

---

## 🚀 Section 3: Applications of Deep CNNs

The practical notebooks demonstrate the application of CNNs to tasks of increasing complexity, from simple digits to complex facial attributes.

### 1. CNN for MNIST Classification (Notebook: `MNIST_with_CNN.ipynb`)

* **First Complete CNN:** This notebook provides the first working implementation of a CNN, demonstrating how to structure the sequential blocks of convolution, pooling, and activation.
* **Architecture:** The model uses several $\text{Conv2d} \rightarrow \text{ReLU} \rightarrow \text{MaxPool2d}$ blocks followed by a **Flatten** layer to transition to the final fully connected classification head.
* **Performance:** The CNN significantly outperforms the Chapter 13 MLP on the MNIST dataset, confirming the effectiveness of the architecture for spatial feature extraction.

### 2. Deep CNN for Facial Attribute Classification (Notebook: `CelebA_with_CNN.ipynb`) 🧑‍🎤

* **Complex Data & Binary Task:** This notebook tackles a complex, real-world task using the **CelebA (Celebrity Attributes)** dataset to perform **Binary Classification** (e.g., detecting if an attribute like 'Smiling' is present).
* **Advanced Data Pipeline:** It requires a robust pipeline for handling large **3-channel (RGB) images** and includes essential preprocessing like **channel-wise Normalization** (`transforms.Normalize`).
* **Binary Prediction:** The network is trained using **`nn.BCEWithLogitsLoss`** and the final layer has 1 output unit.
* **Prediction Visualization:** The notebook visualizes the model's output by displaying the face image alongside the **Ground Truth (GT)** label and the model's **predicted probability** (e.g., `Pr(Smile)=95%`), showcasing high-confidence attribute detection.