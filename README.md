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

**What is Visualization?**  
Visualization in the context of machine learning refers to the graphical representation of data, model performance, or results. It can involve plotting data distributions, learning curves, confusion matrices, Grad-CAM heatmaps, or other metrics to help interpret how a model is behaving and to explore the underlying dataset.

**Why we use Visualization:**  
- **Understanding Data**: Visualization helps in identifying patterns, correlations, and anomalies in the dataset. For example, plotting class distributions can reveal imbalances, and scatter plots can help understand relationships between variables.
- **Model Performance Evaluation**: We use visualization to interpret how well the model is learning, using tools like accuracy and loss plots, confusion matrices, and ROC curves to observe model progress over epochs.
- **Debugging**: Visualization can highlight where the model might be going wrong. For example, Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps can show which parts of an image the model focused on when making a decision, helping us understand and interpret predictions.
- **Communicating Results**: Visualizations make it easier to communicate results to stakeholders, especially non-technical individuals, by providing a clear and concise way to understand the model's output and its reliability.

**What is Imbalanced Data?**  

Imbalanced data refers to a dataset where the distribution of classes is not uniform. In other words, some classes have significantly more samples than others. For example, in a binary classification problem where one class represents 90% of the data and the other class only 10%, the dataset is considered imbalanced. This can lead to problems in training machine learning models, as they may become biased towards the majority class.

As we will see in the next figure:

![Data Visualization](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/Image/Data%20Visualization.png)
