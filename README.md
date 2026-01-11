# Ocular Disease Recognition using Deep Learning

This project implements a **multi-label ocular disease recognition system**
using retinal fundus images and deep learning. The system predicts the
probability of multiple ocular diseases and provides **explainable AI
visualizations using Grad-CAM**.

⚠️ This project is intended for **educational and research purposes only**
and is **not a medical diagnostic tool**.

---

## Dataset

- **Dataset Name:** Ocular Disease Intelligent Recognition (ODIR-5K)
- **Source:** Kaggle  
  https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
- **Description:**  
  The dataset contains retinal fundus images collected from real clinical
  environments, along with multi-label disease annotations provided by
  professional ophthalmologists.

- **Disease Labels:**
  - Normal (N)
  - Diabetes (D)
  - Glaucoma (G)
  - Cataract (C)
  - Age-related Macular Degeneration (A)
  - Hypertension (H)
  - Pathological Myopia (M)
  - Other Abnormalities (O)

---

## Models Implemented

The following deep learning models were trained and compared:

- ResNet-50  
- DenseNet-121  
- EfficientNet-B0 (**final selected model**)

EfficientNet-B0 demonstrated the **best generalization performance**
based on validation accuracy and loss.

---

## Explainability (Grad-CAM)

- Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize
  retinal regions that most influenced each disease prediction.
- Explanations are generated **per disease** to support the multi-label nature
  of ocular conditions.

---

## User Interface

A **Streamlit-based graphical interface** was developed to:
- Upload retinal fundus images
- Display disease probability estimates
- Visualize Grad-CAM explanations directly in the UI

---

## How to Run the Application

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
