# ✨ CHAT - CONJUNCTIVAL HEMOGLOBIN ASSESSMENT TOOL

The **Anemia Detection System** is an AI-powered diagnostic tool developed to quickly, accurately, and non-invasively classify the severity of anemia by analyzing real-time images of the patient's eye conjunctiva.  
This solution significantly enhances accessibility and affordability for early detection compared to traditional invasive blood tests.

---

## 🔑 Key Features

- **Non-Invasive Diagnostics:** Eliminates the need for blood sampling by relying solely on visual analysis of the eye.  
- **Deep Learning Pipeline:** Utilizes a multi-stage flow combining YOLOv8 (Detection), U-Net (Segmentation), and MobileNetV3 (Classification).  
- **Real-Time Processing:** Optimized for quick performance to provide instant diagnostic feedback, suitable for clinical environments.  
- **Multi-Class Classification:** Accurately categorizes anemia into four severity levels: *No Anemia*, *Mild Anemia*, *Moderate Anemia*, and *Severe Anemia*.  
- **High Clinical Accuracy:** Achieved a final validation accuracy of **94.75%** on a real-world clinical dataset.

---

## 💻 Tech Stack

| **Category** | **Technology/Model** | **Purpose** |
|---------------|----------------------|--------------|
| Language | Python (3.x) | Core implementation language |
| Deep Learning | TensorFlow / Keras | Framework for building and training neural networks |
| Image Processing | OpenCV | Used for real-time camera access and basic image manipulation |
| Object Detection | YOLOv8 | Detects and isolates the eye region in preprocessing |
| Segmentation | U-Net (with ResNet-34 Encoder) | Extracts the precise conjunctiva region |
| Classification | MobileNetV3-Large (Fine-tuned) | Final model for severity prediction |
| API/Integration | FastAPI / Python | REST API service for model deployment |

---

## 🧠 Core System Flow and Architecture

The system’s strength lies in its **modular pipeline**, ensuring only the most relevant visual data (the conjunctiva) is passed to the classifier.

---

## 🖼️ Project Execution Flow

The flow diagram illustrates the end-to-end execution, including the **critical feedback loop** where poor-quality images are filtered and the user is prompted to retake the shot.

![Architecture.jpg](Assets/Architecture.jpg)

---

## 👁️ Image Preprocessing 

This critical phase ensures image quality is maintained and standardized before analysis:

- **Eye Detection Model (YOLOv8):** The captured image is first processed by the YOLOv8 model to automatically locate and isolate the main eye region, cropping out irrelevant background data.  
- **Cropping and Resizing:** After detection and segmentation, images are subjected to precise cropping (using contour detection) and resized to a uniform dimension (e.g., 224×224 pixels).  
- **Filtering:** Additional filters (Brightness, Background, Boundary) are applied to enhance color contrast and adjust lighting conditions.

---

## ✂️ Step 1: Segmentation Model (U-Net Architecture)

This model focuses on isolating the conjunctiva from surrounding noise (eyelashes, reflections) using its robust encoder-decoder structure.

The **U-Net** model uses a **ResNet-34 backbone** in the encoder path for powerful feature extraction and skip connections to maintain high-resolution spatial details, resulting in a precise conjunctiva mask.

![Segmentation_Model](Assets/Segmentation_Model.png)

## 🎯 Step 2: Classification Model (MobileNetV3Large Fine-tuned)

The segmented and resized conjunctiva image is classified by the **MobileNetV3-Large** model, chosen for its efficiency and transfer learning capabilities.

![Classification_Model](Assets/Classification_Model.jpg)

The pre-trained MobileNetV3-Large model is fine-tuned on the project’s clinical dataset to accurately predict one of the four anemia categories based on conjunctiva pallor characteristics.

---

## 📊 Performance and Validation

The model’s performance was validated against real-time, clinically labeled data, resulting in robust metrics:

| **Algorithm** | **Accuracy** | **Precision** | **F1 Score** |
|----------------|---------------|----------------|---------------|
| MobileNetV3 (Fine-tuned) | **94.75%** | 0.92 | 0.91 |

---

### Confusion Matrix

The confusion matrix provides clinical context for classification errors and demonstrates model reliability across all four classes.

![Confusion_Matrix.jpg](Assets/Confusion_Matrix.jpg)

---

## ⚙️ Setup and Installation (How to Run)

### 1. Prerequisites
Ensure you have **Python 3.11.4** or higher installed.

### 2. Clone the Repository
```bash
git clone https://github.com/YourUsername/Anemia-Detection-Eye-Conjunctiva-AI.git
cd Anemia-Detection-Eye-Conjunctiva-AI
```

### 3. Install Dependencies
You must install the necessary libraries listed in requirements.txt.

```bash
pip install -r requirements.txt
```

### 4. Setup Models and Data
Place your pre-trained .h5 model files into the ./models/ directory, and ensure your labeled_data.csv is in the ./data/ directory.

### 5. Run the End-to-End Pipeline
Execute the main script to start the system.

```bash
python src/main_pipeline.py
```

This command initiates the OpenCV camera capture and runs the image through the complete detection flow.
