# Chest Disease Detection by X-ray 🩻

### **What is Chest Disease Detection by X-ray?** 🤔

Chest disease detection by X-ray utilizes deep learning models to analyze chest X-ray images for diagnosing conditions such as pneumonia, tuberculosis, and COVID-19. During the COVID-19 pandemic, this became a vital tool, as X-rays provided quick, non-invasive diagnostics. This project focuses on developing a web-based application that allows users to upload X-rays and receive predictions about their lung health, thereby reducing hospital visits and assisting healthcare professionals with quicker diagnoses. The goal is to help both patients and doctors understand whether a person has a respiratory disease, such as:

- **Normal**: No disease. ✅
- **Pneumonia-Bacterial**: Bacterial infection in the lungs. 🦠  
- **Pneumonia-Viral**: Viral infection causing lung inflammation. 🤒  
- **Tuberculosis**: A bacterial infection affecting the lungs. 🏥  
- **Lung Opacity**: Other lung conditions detectable via X-rays. 🌫️  
- **COVID-19**: Viral infection primarily affecting the respiratory system. 🦠🌍

---

### **Datasets Used in This Project** 📊

Three different datasets were utilized for the project, each focusing on detecting various chest diseases:

1. **Dataset 1: COVID-19 Radiography Database (COVID-19 & Pneumonia)** 
   - **Normal**: 10,200 images ✅
   - **Viral Pneumonia**: 1,345 images 🦠  
   - **COVID**: 3,616 images 🦠  
   - **Lung Opacity**: 6,012 images 🌫️  
   - **Bacterial Pneumonia**: 0 images ❌  
   - **Tuberculosis**: 0 images ❌  

2. **Dataset 2: Curated Chest X-Ray Image Dataset for COVID-19** 
   - **Normal**: 3,270 images ✅  
   - **Viral Pneumonia**: 1,656 images 🦠  
   - **COVID**: 1,281 images 🦠  
   - **Lung Opacity**: 0 images ❌  
   - **Bacterial Pneumonia**: 3,001 images 🦠  
   - **Tuberculosis**: 0 images ❌  

3. **Dataset 3: Tuberculosis X-ray** 
   - **Normal**: 514 images ✅  
   - **Viral Pneumonia**: 0 images ❌  
   - **COVID**: 0 images ❌  
   - **Lung Opacity**: 0 images ❌  
   - **Bacterial Pneumonia**: 0 images ❌  
   - **Tuberculosis**: 2,494 images 🦠  

---

### **Why We Use X-rays Instead of PCR?** 💡

X-rays provide a non-invasive, quick, and widely available method for diagnosing lung diseases. While PCR tests are commonly used for detecting viral infections like COVID-19, they do not provide information about the state of the lungs. X-rays, on the other hand, allow medical professionals to visualize lung damage or inflammation, which can indicate diseases such as pneumonia or COVID-19. This helps in a broader diagnosis, especially for conditions that PCR cannot detect. 

---
### **Visualization** 📊

**What is Visualization?** 🖼️  
Visualization in the context of machine learning refers to the graphical representation of data, model performance, or results. It can involve plotting data distributions, learning curves, confusion matrices, Grad-CAM heatmaps, or other metrics to help interpret how a model is behaving and to explore the underlying dataset.

**Why We Use Visualization:** 🔍  
- **Understanding Data**: Visualization helps in identifying patterns, correlations, and anomalies in the dataset. For example, plotting class distributions can reveal imbalances, and scatter plots can help understand relationships between variables.
- **Model Performance Evaluation**: We use visualization to interpret how well the model is learning, using tools like accuracy and loss plots, confusion matrices, and ROC curves to observe model progress over epochs.
- **Debugging**: Visualization can highlight where the model might be going wrong. For example, Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps can show which parts of an image the model focused on when making a decision, helping us understand and interpret predictions.
- **Communicating Results**: Visualizations make it easier to communicate results to stakeholders, especially non-technical individuals, by providing a clear and concise way to understand the model's output and its reliability.

**What is Imbalanced Data?** ⚖️  
This imbalance can cause challenges when training machine learning models, as models may become biased toward the majority class. They might achieve high overall accuracy by primarily predicting the majority class while failing to effectively recognize the minority class. This can lead to poor performance, especially for applications where the minority class is of significant interest, such as in medical diagnoses or fraud detection.


**Class Distribution** 📊  
Here’s the distribution of images across different classes, sorted in descending order by the number of images:

| Class                     | Number of Images | Percentage (%) |
|---------------------------|------------------|----------------|
| Normal                    | 6500             | 25.1           |
| Lung Opacity              | 6012             | 23.2           |
| COVID-19                  | 4897             | 18.9           |
| Pneumonia-Bacterial       | 3001             | 11.6           |
| Pneumonia-Viral           | 3001             | 11.6           |
| Tuberculosis              | 2494             | 9.6            |
| **Total**                 | **25905**        | **100.0**      |

As we will see in the next figure:

![Data Visualization](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/Image/Data%20Visualization.png)

---
### **How Imbalanced Data Can Affect Our Model** ⚖️

Imbalanced data can cause several issues in model performance:

- **Poor Generalization**: The model may struggle to recognize the minority class effectively, leading to high accuracy but poor recall for that class.
- **Overfitting**: The model may focus on the majority class and fail to generalize well to the minority class, resulting in overfitting.
- **Bias**: Predictions may be biased toward the majority class, which can lead to incorrect conclusions, especially in critical applications like medical diagnostics.

### **Solving Data Imbalance Using Augmentation** 🌱

Imbalanced data is a common issue in machine learning, particularly in medical imaging datasets. To address this, **data augmentation** is an effective approach.

#### **What is Data Augmentation?**  
Data augmentation involves creating additional training data from the existing dataset by applying transformations to original images. This technique helps balance the dataset by artificially increasing the number of samples in underrepresented classes.

#### **Why Use Augmentation for Imbalanced Data?**  
- **Increasing Diversity**: Introduces variability in the dataset, allowing the model to generalize better.
- **Balancing Class Distribution**: Augmenting only the minority classes creates a more balanced dataset without the need for additional real-world data.
- **Improving Model Robustness**: Augmented images make the model more resilient to variations like rotations, flips, and other distortions, enhancing performance on unseen data.

### **Common Augmentation Techniques**  
Here are some commonly used techniques to handle data imbalance:

1. **Rotation**: Rotating images by small angles to create new perspectives.
2. **Zooming**: Randomly zooming into images to simulate different scales.
3. **Shifting**: Translating images along the x or y axis.
   ```python
   from keras.preprocessing.image import ImageDataGenerator
   
   train_gen = ImageDataGenerator(
       rescale=1. / 255,
       zoom_range=0.1,
       width_shift_range=0.1,
       height_shift_range=0.1,
       rotation_range=10,
       fill_mode='nearest'
   )
   test_gen = ImageDataGenerator(rescale=1. / 255)
   val_gen = ImageDataGenerator(rescale=1. / 255)
---
### **Structure of the Datasets** 📂

Each dataset follows the typical data structure for deep learning tasks:

- **Training Set**: 🏋️‍♂️ This set is used to train the model and constitutes about 70-80% of the dataset.
- **Validation Set**: 🔧 This is 10-20% of the dataset and is used to fine-tune model hyperparameters during training to avoid overfitting.
- **Test Set**: 🧪 This set, around 10-20% of the dataset, is used to evaluate the model’s performance on unseen data.

---

### **What is Train, Validation, and Test Data?** 🧠

- **Training Data**: 📘 This is the portion of the dataset used by the model to learn patterns and make predictions.
- **Validation Data**: 🧐 This set is used during training to validate the model’s performance and adjust hyperparameters.
- **Test Data**: 🎯 This final set is used to evaluate the performance of the fully trained model on unseen data.

---

### **Why Do We Split Data into Train, Validation, and Test?** 🤔

The purpose of splitting the data is to ensure the model generalizes well to new, unseen data. Here's why we split the data:

- **Training Data**: 🛠️ Teaches the model to recognize patterns.
- **Validation Data**: 🛎️ Helps tune the model's hyperparameters and check performance during training.
- **Test Data**: 📊 Provides an unbiased evaluation of the model's performance after training is complete.

---
### **What is VGG19?** 🤖

VGG19 is a deep convolutional neural network model with 19 layers, developed by the Visual Geometry Group (VGG) at Oxford University. It was pre-trained on the ImageNet dataset, which contains over a million images across 1000 different classes. VGG19 is known for its simple and uniform structure, using small (3x3) convolution filters but with a large depth.

---

### **How Similar is VGG19 ImageNet with Chest Diseases Images?** 🩺

VGG19 was originally trained on ImageNet, which includes general images like animals, plants, and objects. While these images are very different from medical images like X-rays, transfer learning allows VGG19 to apply the knowledge it learned from ImageNet to medical imaging tasks. By fine-tuning the final layers, VGG19 can focus on the specific features relevant to chest disease detection.

---

### **Advantages and Disadvantages of VGG19** ⚖️

- **Advantages**:
  - **Proven Performance**: VGG19 is well-known for its accuracy in image classification tasks.
  - **Transfer Learning**: It allows for faster training on medical images by leveraging pre-trained weights from ImageNet.

- **Disadvantages**:
  - **Computationally Expensive**: Requires significant processing power and memory, especially for large datasets like X-ray images.
  - **Slow Training Time**: The large depth of the network can result in slow training.

---

### **Why We Chose VGG19 Over VGG16** 🔄

Initially, we tried using **VGG16** for our chest disease classification, but the **accuracy** results were not as high as we expected. After exploring different models, we chose **VGG19** because its deeper architecture and feature extraction capabilities are more suitable for medical imaging tasks like chest X-rays. Despite the limited resources, we managed to achieve good accuracy using VGG19, and we know that with better computational power, the accuracy could be significantly improved.
