# Secure AI on MNIST – Adversarial & Poisoning Experiments (Tasks 5, 6, 7)

This project implements the **Secure AI MNIST** assignment focusing on:

- **Task 5 – Data Poisoning**
  - **Method 1:** Trigger-based poisoning with a small corner square.
  - **Method 2:** Adversarial examples using **FGSM** (Fast Gradient Sign Method) via **ART**.
- **Task 6 – Testing With New Test Data**
  - Evaluate the CNN on **clean**, **triggered**, and **FGSM adversarial** test sets.
- **Task 7 – Protection (Blue Teaming)**
  - Train a **defense model** using a mix of clean and FGSM-perturbed training samples.

The code is designed for **Jupyter / Google Colab**, but can also be adapted to a plain Python script.

---

## 1. Project Overview

### 1.1. Dataset

We use **MNIST** (handwritten digits 0–9):

- 60,000 training images (28×28 grayscale)
- 10,000 test images

Images are normalized to the range **[0, 1]** and reshaped to `(28, 28, 1)`.

### 1.2. Base CNN Architecture

A simple CNN built with **TensorFlow / Keras**:

- `Conv2D(32, 3×3, ReLU, same)`
- `MaxPooling2D(2×2)`
- `Conv2D(64, 3×3, ReLU, same)`
- `MaxPooling2D(2×2)`
- `Flatten`
- `Dense(128, ReLU)`
- `Dropout(0.3)`
- `Dense(10, softmax)`

Optimizer: **Adam (1e-3)**  
Loss: **categorical cross-entropy**  
Metrics: **accuracy**

This CNN is trained first on **clean MNIST** and reused throughout all experiments.

---

## 2. Methods Implemented

### 2.1. Method 1 – Trigger-Based Data Poisoning

- Choose ~100 training samples of a specific digit (e.g. **digit 7**).
- Add a small **white square** in a corner of the image (e.g. bottom-right).
- This creates a **poisoned training subset** (used for visualization).
- For evaluation, we construct a **triggered test set** by stamping the same square on **every test image**.

> In this setup we *do not* change the labels of the poisoned samples; the trigger tests how such a pattern affects performance even when labels remain the same.

**Outputs:**

- Example visualization of clean vs poisoned digits:
  - `secure_ai_outputs/images/method1_poison_examples.png`
- Triggered test set is used for metrics under **`triggered_test_method1`** in the summary JSON.

---

### 2.2. Method 2 – FGSM Adversarial Examples (ART)

We use **Adversarial Robustness Toolbox (ART)** to generate adversarial images:

- Wrap the trained baseline CNN using `KerasClassifier`.
- Use **FastGradientMethod** (FGSM) with:
  - `eps = 0.3`
  - `batch_size = 128`
- Generate:
  - **FGSM training set** (`x_train_adv`) – used later for defense training (Task 7).
  - **FGSM test set** (`x_test_adv`) – used as “new test data” for Task 6.

**Outputs:**

- Visualization of clean vs adversarial test images:
  - `secure_ai_outputs/images/method2_clean_vs_fgsm.png`
- Metrics for FGSM test set under **`fgsm_test_method2`** in the summary JSON.

---

### 2.3. Task 7 – Protection / Blue Teaming (FGSM Adversarial Training)

To improve robustness:

1. **Mix** clean and adversarial training data:
   - Concatenate `x_train` (clean) and `x_train_adv` (FGSM).
   - Labels are duplicated from the clean set.
   - Shuffle the combined dataset.
2. Train a **new CNN** (same architecture) on the mixed data.
3. Evaluate this **defense model** on:
   - Clean test set  
   - FGSM test set  

**Outputs:**

- Confusion matrices for the defense model:
  - `secure_ai_outputs/images/defense_confusion_clean.png`
  - `secure_ai_outputs/images/defense_confusion_fgsm.png`
- Metrics under **`defense_fgsm.clean_test`** and **`defense_fgsm.fgsm_test`** in the summary JSON.
- Model weights:
  - `secure_ai_outputs/models/cnn_mnist_defense_fgsm.keras`

---

## 3. Metrics

For each evaluation, we record **only**:

- **Loss**
- **Accuracy**
- **Confusion matrix**
- **Inference time**
  - Total time for evaluating the entire test set.
  - Time per sample (ms/sample).

There are **no classification reports** in the console output, and confusion matrices are **only** shown as annotated heatmaps.

### 3.1. Where Metrics Are Stored

All metrics are summarized in:

```text
secure_ai_outputs/metrics_summary_tasks5_6_7.json
```

Structure (example):

```json
{
  "clean_test": {
    "loss": 0.0234,
    "accuracy": 0.9923,
    "inference_time_s": 0.45,
    "inference_time_per_sample_ms": 0.045,
    "confusion_matrix": [[...], [...], ...]
  },
  "triggered_test_method1": {
    "loss": ...,
    "accuracy": ...,
    "inference_time_s": ...,
    "inference_time_per_sample_ms": ...,
    "confusion_matrix": [[...], ...]
  },
  "fgsm_test_method2": {
    "loss": ...,
    "accuracy": ...,
    "inference_time_s": ...,
    "inference_time_per_sample_ms": ...,
    "confusion_matrix": [[...], ...]
  },
  "defense_fgsm": {
    "clean_test": {
      "loss": ...,
      "accuracy": ...,
      "inference_time_s": ...,
      "inference_time_per_sample_ms": ...,
      "confusion_matrix": [[...], ...]
    },
    "fgsm_test": {
      "loss": ...,
      "accuracy": ...,
      "inference_time_s": ...,
      "inference_time_per_sample_ms": ...,
      "confusion_matrix": [[...], ...]
    }
  }
}
```

---

## 4. Outputs & Directory Structure

After running the main script/notebook, you should have:

```text
secure_ai_outputs/
├─ images/
│  ├─ method1_poison_examples.png
│  ├─ method2_clean_vs_fgsm.png
│  ├─ baseline_confusion_clean.png
│  ├─ baseline_confusion_triggered.png
│  ├─ baseline_confusion_fgsm.png
│  ├─ defense_confusion_clean.png
│  └─ defense_confusion_fgsm.png
├─ models/
│  ├─ cnn_mnist_baseline.keras
│  └─ cnn_mnist_defense_fgsm.keras
└─ metrics_summary_tasks5_6_7.json
```

---

## 5. How to Run

### 5.1. Requirements

- Python 3.8+  
- Recommended: virtual environment (venv/conda)

Core libraries:

- `tensorflow`
- `numpy`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `adversarial-robustness-toolbox` (ART)

### 5.2. Installation

```bash
# Clone your repository
git clone <YOUR_REPO_URL>.git
cd <YOUR_REPO_NAME>

# (Optional) create and activate virtual env
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scriptsctivate

# Install dependencies
pip install tensorflow numpy matplotlib pandas scikit-learn adversarial-robustness-toolbox
```

### 5.3. Running in Jupyter / Colab

1. Open your notebook (e.g. `SecureAI_MNIST.ipynb`) in Jupyter / Colab.
2. Run the single large cell starting with:

   ```python
   # ============================================================
   # Secure AI MNIST – Tasks 5, 6, 7
   # ...
   ```

3. Wait for training and evaluation to finish.
4. Check:
   - **Console output** for loss/accuracy/inference-time logs.
   - **`secure_ai_outputs/images/`** for plots.
   - **`secure_ai_outputs/models/`** for saved models.
   - **`secure_ai_outputs/metrics_summary_tasks5_6_7.json`** for metrics summary.

### 5.4. Running as a Script (Optional)

You can also wrap the cell contents into a Python file, e.g. `secure_ai_mnist.py`, and run:

```bash
python secure_ai_mnist.py
```

(Ensure the working directory is the repo root so `secure_ai_outputs/` is created correctly.)

---

## 6. Interpreting the Results

### 6.1. Baseline vs Triggered vs FGSM

- **`clean_test`**  
  - Baseline performance on unmodified MNIST.
- **`triggered_test_method1`**  
  - Performance when a simple corner pattern (backdoor trigger) is added to each test image.
- **`fgsm_test_method2`**  
  - Performance on images slightly perturbed by FGSM to cause misclassification.

Compare accuracies and confusion matrices to see how fragile the model is to:

- **Backdoor-like triggers** (simple deterministic pattern).
- **Gradient-based adversarial perturbations**.

### 6.2. Defense Model (Blue Teaming)

- **`defense_fgsm.clean_test`** vs **`clean_test`**  
  → How much clean accuracy is retained after adversarial training.
- **`defense_fgsm.fgsm_test`** vs **`fgsm_test_method2`**  
  → How much robustness we gain against FGSM attacks by training on clean + FGSM data.

---

## 7. Non-Code Tasks (for the Report)

The original assignment likely also includes:

- **Threat Modeling (STRIDE)**  
  – Discuss threats to this ML pipeline: spoofing, tampering (data poisoning), information disclosure, etc.
- **Static Analysis Security Testing (SAST)**  
  – Run tools such as `bandit` or `semgrep` over your Python code and summarize findings.

These are **not implemented in code** here, but you can reference this repository’s scripts and notebooks in your written report when you:

- Explain where the CNN is vulnerable (data, model, interface).
- Show how your FGSM defense acts as a blue-team countermeasure.

---

## 8. Acknowledgements

- **MNIST dataset** – Yann LeCun et al.
- **TensorFlow/Keras** – for deep learning.
- **Adversarial Robustness Toolbox (ART)** – for generating FGSM adversarial examples.

---

If you are reviewing this on GitHub, the key files to explore are:

- Main notebook/script implementing tasks 5, 6, 7.
- `secure_ai_outputs/images/` – all visual results.
- `secure_ai_outputs/models/` – trained models.
- `secure_ai_outputs/metrics_summary_tasks5_6_7.json` – final metric summary.
