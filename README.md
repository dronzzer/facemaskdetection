# Facemask detection model 
# Table of Contents
- [Introduction](#Introduction)
- [Features](#Features)
- [Installation](#Installtion)
- [Dataset](#Dataset)
- [Usage](#Usage)


# Introduction
This project is a deep learning-based face mask detection model. It aims to identify whether a person is wearing a mask or not in real-time. The model uses neural networks to classify images and provides accurate and efficient mask detection, which can be utilized in public places to ensure safety protocols are followed.

# Features
- Real-time face mask detection
- High accuracy with deep learning neural networks
- Easy integration with webcam and other video sources
- Customizable for different environments and use cases

# Installation
To set up the project locally, follow these steps:
Clone the repository:
```
git clone https://github.com/yourusername/mask-detection.git
```
Navigate to the project directory:
```
cd mask-detection
```
Create a virtual environment:
```
python -m venv venv
```
Activate the virtual environment:
On Windows:
```
venv\Scripts\activate
```
On macOS and Linux:
```
source venv/bin/activate
```
Install the required dependencies:
```
pip install -r requirements.txt
```

# Usage
get the dataset from https://www.kaggle.com/datasets/belsonraja/face-mask-dataset-with-and-without-mask?resource=download-directory&select=facemask-dataset.

##  Data Preperation 
Organize your dataset in the following structure:
```
dataset/
├── with_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── without_mask/
    ├── image1.jpg
    ├── image2.jpg
    └── ...

```
Change the Location of the dataset in the 'Data preperation.py' script according location at which dataset dirctory is stored.

![2024-08-04](https://github.com/user-attachments/assets/3e450534-eb56-4766-a844-e52748014192)

Run the data preparation script to preprocess the images:
```
python "Data preparation.py"
```

## Training the Model
Train the model using the prepared data:
```
python "Model Training.py"
```
The trained model will be saved as model.keras.

## Real-Time Classification
Run the real-time classification script:
```
python "Real time classification.py"
```
The script will start the webcam and display a live feed with face mask detection.
