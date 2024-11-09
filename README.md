# Weed Detection in Crop Fields with YOLOv7

## Overview
This project aims to develop a computer vision model to detect weeds in crop fields, allowing a robot to spray herbicides only in targeted areas. By minimizing unnecessary pesticide use, this approach supports sustainable farming practices and enhances resource efficiency. The model was fine-tuned using Ultralytics YOLOv7 to ensure high precision and reliability in detecting weeds amidst crops.

## What I Learned
In this project, I gained in-depth experience with advanced computer vision techniques, including object detection and model fine-tuning. I also explored data augmentation strategies to improve model robustness and accuracy, particularly for real-time applications in precision agriculture.

## Dataset
### Crop and Weed Detection Dataset
The dataset includes over 5,000 annotated images of crop fields with weed labels, which were used to train and validate the YOLOv7 model.

### Key Features:
- **Image ID**: Unique identifier for each image
- **Bounding Boxes**: Coordinates outlining weeds within each image

## Problem Statement
In agriculture, the presence of weeds in crop fields significantly impacts crop yields and requires extensive pesticide use. Traditional blanket spraying methods increase costs and harm the environment. This project addresses this issue by developing a model that enables selective herbicide application, reducing overall pesticide usage and improving sustainability.

## Methodology

### Data Preparation
- **Data Augmentation**: Applied techniques such as rotation, flipping, and scaling to diversify training images and enhance model generalization.
- **Image Preprocessing**: Images were resized to a fixed dimension suitable for YOLOv7, ensuring consistent input size for the model.
- **Annotation Parsing**: Extracted bounding box annotations to create labeled data for training.

### Model Building
- **Model**: Fine-tuned **Ultralytics YOLOv7** with custom data to detect weed locations in crop fields.
- **Hyperparameter Tuning**: Experimented with learning rates, batch sizes, and anchor boxes to optimize model performance.
- **Data Augmentation**: Employed rotation, flipping, and scaling to increase the dataset’s diversity and improve model generalization.

### Model Evaluation
The model’s performance was assessed through various evaluation metrics to ensure its robustness for real-world deployment:
- **Mean Average Precision (mAP)**: Achieved **mAP of 0.85** at an IoU threshold of 0.5.
- **Precision**: Model reached a precision score of **0.8**, indicating high accuracy in identifying weed locations.
- **Recall**: Emphasized high recall to ensure most weed instances were detected.
  
### Evaluation Metrics
- **Confusion Matrix**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC Curve**

## Results and Insights
The fine-tuned YOLOv7 model demonstrated high accuracy in weed detection, achieving an mAP of 0.85 and precision of 0.8 at an IoU threshold of 0.5. These results suggest that the model is well-suited for selective spraying applications, enabling targeted pesticide use and reducing environmental impact.

## Visualization
- **Feature Importance Plot**: Highlighted the importance of weed localization for accurate predictions.
- **ROC and Precision-Recall Curves**: Assessed model performance and trade-offs, ensuring a balance between precision and recall.
- **Bounding Box Visualizations**: Displayed sample outputs with detected weed locations in crop fields.

## Libraries Used
- **Pandas**: Data manipulation and analysis
- **OpenCV**: Image processing and augmentation
- **Seaborn & Matplotlib**: Data visualization
- **Ultralytics YOLOv7**: Model training and fine-tuning
- **Scikit-learn**: Model evaluation and metrics

## Conclusion
The weed detection model effectively identifies and localizes weeds in crop fields, empowering a robotic system to apply pesticides only where necessary. This approach minimizes chemical usage, promoting more sustainable farming practices. Future improvements could include testing additional deep learning architectures, exploring segmentation methods for even finer weed identification, and deploying the model for real-time field applications.

## Exposure
**Computer Vision, Ultralytics YOLOv7, Object Detection, Data Augmentation, Precision Agriculture, Autonomous Robotics, Pesticide Optimization, Real-time Detection**
