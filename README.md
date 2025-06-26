# Brain Tumor Classification using MATLAB

This MATLAB project performs classification of brain tumors from MRI images using deep learning models. It provides both training and testing capabilities, along with a graphical user interface (GUI) for post-training interaction.

## Overview

- Training and evaluation are done through `main.m`.
- The model is trained using either a custom CNN or GoogleNet.
- After training, users can select test images and perform classification through a user-friendly GUI.
- The GUI includes options to choose the model (CustomCNN or GoogleNet) and select optimization settings such as training intensity (low/high).

## How It Works

1. **Run `main.m`**
   - Load and preprocess the dataset
   - Train the selected model (CustomCNN or GoogleNet)
   - Save the trained model to disk
   - Generate and save performance metrics, including confusion matrices

2. **Use the GUI**
   - Launch the GUI script (`brain_tumor_gui_only.m`)
   - Select a test image for classification
   - Choose the trained model and adjust optimization settings as desired
   - View the classification result and related metrics

## Features

- Supports both lightweight custom CNN and pretrained GoogleNet
- GUI-based image selection and classification
- Model switching and optimization control
- Automatic generation of performance metrics
- MATLAB-based training and inference environment

## Requirements

- MATLAB R2021a or newer
- Deep Learning Toolbox
- Image Processing Toolbox
- Pretrained GoogleNet (optional, if using that model)

## How to Run

### 1. Train the Model
Run the training script:

```matlab
main
```

This will:
- Load and preprocess images from `brain_tumor_dataset/`
- Train the selected model (CustomCNN or GoogleNet)
- Save the model and performance reports

### 2. Use the GUI

Run the GUI script:

```matlab
brain_tumor_gui_only
```

Follow the steps below to classify an MRI image.

---

## Step-by-Step: GUI Usage

### 1. Launch the GUI  
**Action:** Run `brain_tumor_gui_only.m` in MATLAB  
**Description:** Opens the main graphical user interface of the application  

![Launch GUI](photos/10.png)

---

### 2. Load an MRI Image  
**Action:** Click "Load Image"  
**Description:** Choose an MRI image from your computer to classify  

![Select Image](photos/14.png)  
![Select Image](photos/15.png)  
![Select Image](photos/16.png)

---

### 3. Choose Model & Options  
**Action:** Select model type and training intensity  
**Description:** You can choose between CustomCNN or GoogleNet and set training level to low or high  

![Model Selection](photos/11.png)  
![Model Selection](photos/12.png)  
![Model Selection](photos/13.png)

---

### 4. Classify the Image  
**Action:** Click the "Classify" button  
**Description:** The selected model will process the image and make a prediction  

![Classification](photos/20.png)

---

### 5. View Output  
**Action:** Wait for the result to be displayed  
**Description:** Classification label, confidence score, and optional confusion matrix are shown  

![Result](photos/17.png)  
![Result](photos/8.png)


---
## Screenshots Summary

| Step | Description           | Image(s)                                                        |
|------|-----------------------|------------------------------------------------------------------|
| 1    | Launch GUI            | ![1](photos/10.png)                                              |
| 2    | Select Test Image     | ![2a](photos/14.png) ![2b](photos/15.png) ![2c](photos/16.png)   |
| 3    | Choose Model/Options  | ![3a](photos/11.png) ![3b](photos/12.png) ![3c](photos/13.png)   |
| 4    | Classification        | ![4](photos/20.png)                                              |
| 5    | Results               | ![5a](photos/17.png) ![5b](photos/8.png)                         |

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or collaborations, contact:  
**mertcankankilic27@gmail.com**
