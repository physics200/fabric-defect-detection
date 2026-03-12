# Fabric Defect Detection using Deep Learning

This project detects defects in fabric images using a U-Net based deep learning model.

The model learns to identify defective regions in textile images and highlights them automatically.

Project Motivation
While learning computer vision and deep learning, I wanted to build a project that solves a real industrial problem. Fabric defect detection is widely used in textile manufacturing for quality control.

Dataset
DAGM Surface Defect Dataset was used for training.

Features
- Image segmentation using U-Net
- Detects and highlights fabric defects
- Visual defect overlay on original image

Tech Stack
Python  
TensorFlow / Keras  
OpenCV  
NumPy  
Matplotlib

Project Structure
src/train.py – trains the model  
src/predict.py – runs defect detection on images  

Example Output
The model predicts a defect mask and highlights the defect region on the original image.

Future Improvements
- Improve segmentation accuracy
- Build a web interface for defect detection
- Train on larger industrial datasets