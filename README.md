# Classifying Respiratory Diseases from X-rays
 <ul>
        <li><a href="#what-is-chest-disease-detection">What is Chest Disease Detection by X-ray?</a></li>
        <li><a href="#datasets-used">Datasets Used in This Project</a></li>
        <li><a href="#why-use-xrays">Why We Use X-rays Instead of PCR?</a></li>
        <li><a href="#visualization">Visualization</a></li>
        <li><a href="#imbalanced-data">Imbalanced Data</a></li>
        <li><a href="#class-distribution">Class Distribution</a></li>
        <li><a href="#imbalanced-data-impact">Imbalanced Data Can Affect Our Model</a></li>
        <li><a href="#solving-data-imbalance">Solving Data Imbalance</a></li>
        <li><a href="#structure-of-datasets">Structure of the Datasets</a></li>
        <li><a href="#chale">Enhancing Image Contrast with CHALE</a></li>
        <li><a href="#vgg19">Modeling</a></li>
    </ul>
<h2 id="what-is-chest-disease-detection">What is Chest Disease Detection by X-ray? 🤔</h2>

Chest disease detection by X-ray utilizes deep learning models to analyze chest X-ray images for diagnosing conditions such as pneumonia, tuberculosis, and COVID-19. During the COVID-19 pandemic, this became a vital tool, as X-rays provided quick, non-invasive diagnostics. This project focuses on developing a web-based application that allows users to upload X-rays and receive predictions about their lung health, thereby reducing hospital visits and assisting healthcare professionals with quicker diagnoses. The goal is to help both patients and doctors understand whether a person has a respiratory disease, such as:

- **Normal**: No disease. ✅
- **Pneumonia-Bacterial**: Bacterial infection in the lungs. 🦠  
- **Pneumonia-Viral**: Viral infection causing lung inflammation. 🤒  
- **Tuberculosis**: A bacterial infection affecting the lungs. 🏥  
- **Lung Opacity**: Other lung conditions detectable via X-rays. 🌫️  
- **COVID-19**: Viral infection primarily affecting the respiratory system. 🦠🌍

---

 <h2 id="datasets-used">Datasets Used in This Project 📊</h2>
 
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

<h2 id="why-use-xrays">Why We Use X-rays Instead of PCR? 💡</h2>

X-rays provide a non-invasive, quick, and widely available method for diagnosing lung diseases. While PCR tests are commonly used for detecting viral infections like COVID-19, they do not provide information about the state of the lungs. X-rays, on the other hand, allow medical professionals to visualize lung damage or inflammation, which can indicate diseases such as pneumonia or COVID-19. This helps in a broader diagnosis, especially for conditions that PCR cannot detect. 

---
<h2 id="visualization">Visualization 📊</h2>

**What is Visualization?** 🖼️  
Visualization in the context of machine learning refers to the graphical representation of data, model performance, or results. It can involve plotting data distributions, learning curves, confusion matrices, Grad-CAM heatmaps, or other metrics to help interpret how a model is behaving and to explore the underlying dataset.

**Why We Use Visualization:** 🔍  
- **Understanding Data**: Visualization helps in identifying patterns, correlations, and anomalies in the dataset. For example, plotting class distributions can reveal imbalances, and scatter plots can help understand relationships between variables.
- **Model Performance Evaluation**: We use visualization to interpret how well the model is learning, using tools like accuracy and loss plots, confusion matrices, and ROC curves to observe model progress over epochs.
- **Debugging**: Visualization can highlight where the model might be going wrong. For example, Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps can show which parts of an image the model focused on when making a decision, helping us understand and interpret predictions.
- **Communicating Results**: Visualizations make it easier to communicate results to stakeholders, especially non-technical individuals, by providing a clear and concise way to understand the model's output and its reliability.

 <h2 id="class-distribution">Class Distribution 📊</h2>
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

![Data Visualization](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/Dataset.png)

---
 <h2 id="structure-of-datasets">Structure of the Datasets 📂</h2>

Each dataset follows the typical data structure for deep learning tasks:

- **Training Set**: 🏋️‍♂️ This set is used to train the model and constitutes about 70-80% of the dataset.
- **Validation Set**: 🔧 This is 10% of the dataset and is used to fine-tune model hyperparameters during training to avoid overfitting.
- **Test Set**: 🧪 This set, around 10% of the dataset, is used to evaluate the model’s performance on unseen data.

---

<h2 id="chale"> CHALE_Contrast Histogram Adaptive Limited Equalization</h2>


**CHALE** (Contrast Histogram Adaptive Limited Equalization) is a Python-based image enhancement filter designed to improve the contrast of images by performing adaptive histogram equalization with a contrast limit. This approach enhances important details in images without over-amplifying noise, making it ideal for applications like medical imaging where subtle details are crucial.

CHALE's ability to enhance image contrast through Contrast-Limited Adaptive Histogram Equalization sets it apart as an invaluable tool in medical imaging. This unique approach enhances critical visual details in medical images, such as X-rays, by amplifying important contrasts without over-emphasizing noise. By refining the visibility of subtle patterns and structures, CHALE contributes to a more accurate diagnostic process, standing to make a profound impact on patient outcomes.

**Imagine the possibilities:**

-**Highlighting Fine-Grained Features**: CHALE brings out intricate details within medical images, enhancing subtle variations in texture, shape, and opacity that are essential for precise diagnosis.

-**Enhancing Image Clarity Across Cases**: With CHALE’s adaptable contrast enhancement, images from diverse sources can be processed to reveal critical features, making diagnosis more consistent and reliable across varied datasets.

-**Reducing Annotation Burden**: CHALE’s use of contrast enhancement reduces the dependency on extensively labeled datasets, helping healthcare professionals work more efficiently, even with limited annotations.

---

### Comparison of CHALE-Enhanced Images with Original Images 🖼️

The CHALE filter enhances X-ray images by improving contrast, making it easier to identify and analyze various respiratory conditions. Below is a comparison showcasing the original images alongside the enhanced CHALE images.

| Condition                | Comparison of CHALE-Enhanced Images and Original Images                      |
|-------------------------|---------------------------------------------------------|
| **COVID-19**            | ![CHALE Image 1](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/CLAHE_COVID-19.png)   |
| **Normal**              | ![CHALE Image 2](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/CLAHE_Normal.png)   |
| **Lung Opacity**        | ![CHALE Image 3](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/CLAHE_Lung_Opacity.png)   |
| **Pneumonia-Bacterial** | ![CHALE Image 4](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/CLAHE_Pneumonia-Bacterial.png)   |
| **Pneumonia-Viral**     | ![CHALE Image 5](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/CLAHE_Pneumonia-Viral.png)   |
| **Tuberculosis**        | ![CHALE Image 6](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/CLAHE_Tuberculosis.png)   |


### Key Differences

1. **Enhanced Contrast**: The CHALE-transformed images unveil lung opacities and inflammations that are critical in diagnosing diseases like **pneumonia** or **COVID-19**. This clarity can make all the difference in timely and effective treatment.
2. **Clearer Edges and Boundaries**: CHALE sharpens the edges of tissues, illuminating the distinctions between healthy and affected areas. This precision is vital in identifying conditions like **tuberculosis**, where visible lesions can mean life or death.
3. **Improved Feature Recognition**: CHALE empowers deep learning models to discern textures and densities that may be overlooked. This capability is crucial for differentiating between conditions, such as **viral versus bacterial pneumonia**, ensuring patients receive the most accurate diagnoses.

---
<h2 id="imbalanced-data">Imbalanced Data ⚖️</h2>
This imbalance can cause challenges when training machine learning models, as models may become biased toward the majority class. They might achieve high overall accuracy by primarily predicting the majority class while failing to effectively recognize the minority class. This can lead to poor performance, especially for applications where the minority class is of significant interest, such as in medical diagnoses or fraud detection.
---
<h2 id="imbalanced-data-impact">Imbalanced Data Can Affect Our Model ⚖️</h2>

Imbalanced data can cause several issues in model performance:

- **Poor Generalization**: The model may struggle to recognize the minority class effectively, leading to high accuracy but poor recall for that class.
- **Overfitting**: The model may focus on the majority class and fail to generalize well to the minority class, resulting in overfitting.
- **Bias**: Predictions may be biased toward the majority class, which can lead to incorrect conclusions, especially in critical applications like medical diagnostics.
- 


<h2 id="solving-data-imbalance">Solving Data Imbalance Using Augmentation And Class Weights 🌱</h2>

<p>Imbalanced data is a common issue in machine learning, particularly in medical imaging datasets. To address this, <strong>data augmentation</strong> is an effective approach, and
<b>Class weights</b> help to give more importance to the minority classes, ensuring that the model pays adequate attention to all classes during training.</p>

---

#### **What is Data Augmentation?**  
Data augmentation involves creating additional training data from the existing dataset by applying transformations to original images. This technique helps balance the dataset by artificially increasing the number of samples in underrepresented classes.

#### **Why Use Augmentation for Imbalanced Data?**  
- **Increasing Diversity**: Introduces variability in the dataset, allowing the model to generalize better.
- **Balancing Class Distribution**: Augmenting only the minority classes creates a more balanced dataset without the need for additional real-world data.
- **Improving Model Robustness**: Augmented images make the model more resilient to variations like rotations, flips, and other distortions, enhancing performance on unseen data.
---
#### **Why Use Class Weights Work?**
The class weights are calculated based on the distribution of the classes in the training data. By assigning a higher weight to the minority classes, the model will be penalized more for misclassifying these samples, encouraging it to learn more about them.

#### **Class Weights Implementation**
While we focus on rescaling our images, we will use the following approach to compute class weights:

```python
from sklearn.utils.class_weight import compute_class_weight

# Get the class indices from the training generator
class_indices = train_generator.class_indices
class_labels = list(class_indices.keys())

# Obtain the labels for each image in the training data
train_labels = train_generator.classes

# Compute the class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = dict(enumerate(class_weights))

print("Class Labels:", class_labels)
print("Class Weights:", class_weights_dict)
```
---
 <h2 id="vgg19">What is VGG19? 🤖</h2>

VGG19 is a deep convolutional neural network model with 19 layers, developed by the Visual Geometry Group (VGG) at Oxford University. It was pre-trained on the ImageNet dataset, which contains over a million images across 1000 different classes. VGG19 is known for its simple and uniform structure, using small (3x3) convolution filters but with a large depth.

---

<h2 id="#similarity-vgg19-chest-diseases">How Similar is VGG19 ImageNet with Chest Diseases Images? 🩺</h2>

VGG19 was originally trained on ImageNet, which includes general images like animals, plants, and objects. While these images differ from medical images like X-rays, the model demonstrated a high level of similarity with the X-ray data during transfer learning. This allowed VGG19 to effectively apply the knowledge it learned from ImageNet to medical imaging tasks. By fine-tuning the final layers, VGG19 was able to focus on the specific features relevant to chest disease detection, improving its performance on X-ray data.

---

<h2 id ="#vgg19-advantages-disadvantages">Advantages and Disadvantages of VGG19 ⚖️</h2>

- **Advantages**:
  - **Proven Performance**: VGG19 is well-known for its accuracy in image classification tasks.
  - **Transfer Learning**: It allows for faster training on medical images by leveraging pre-trained weights from ImageNet.

- **Disadvantages**:
  - **Computationally Expensive**: Requires significant processing power and memory, especially for large datasets like X-ray images.
  - **Slow Training Time**: The large depth of the network can result in slow training.

---

 ## Why We Chose VGG19 Over VGG16 🔄
Initially, we tried using **VGG16** for our chest disease classification, but the accuracy results were not as high as we expected. After exploring different models, we chose **VGG19** because its deeper architecture and feature extraction capabilities are more suitable for medical imaging tasks like chest X-rays. Despite the limited resources, we managed to achieve good accuracy using VGG19, and we know that with better computational power, the accuracy could be significantly improved.

## Model Comparison: VGG19 vs VGG16 vs VGG19 with CHALE vs From Scratch 📊
Here, we showcase the performance of **VGG19**, **VGG19 with CHALE**, **VGG16**, and a **model trained from scratch**. The results highlight the impact of using CHALE for enhancing the image processing capabilities of VGG19.

### Performance Summary
- **VGG19 with CHALE** achieved the highest accuracy at **97.46%**, showcasing the effectiveness of the CHALE technique in improving model performance.
- **VGG19** followed with an accuracy of **95.54%**.
- **From Scratch** achieved an accuracy of **91.91%**.
- **VGG16** had the lowest accuracy at **80.02%**, demonstrating the need for deeper architectures in medical imaging tasks.

### Detailed Model Performance
| **Model**                | **Train Loss** | **Val Loss** | **Test Loss** | **Train Accuracy** | **Val Accuracy** | **Test Accuracy** |
|--------------------------|----------------|--------------|---------------|--------------------|------------------|-------------------|
| **VGG19 with CHALE**     | **0.0681**     | **0.2247**   | **0.2257**    | **97.46%**          | **92.24%**        | **92.56%**         |
| **VGG19**                | 0.1166         | 0.2392       | 0.2365        | **95.54%**          | **91.63%**        | **91.65%**         |
| **From Scratch**         | 0.9191         | 0.8634       | 0.8587        | **91.91%**          | **86.34%**        | **85.87%**         |
| **VGG16**                | 0.4927         | 0.5423       | 0.5021        | **80.02%**          | **77.61%**        | **79.89%**         |

The results illustrate that the use of **CHALE** with **VGG19** not only improved the accuracy but also enhanced the model's ability to process images effectively, making it a valuable addition for medical imaging tasks.

### **VGG19 with CHALE Loss and Accuracy**:
- Here is a visualization of **VGG19 with CHALE's Loss and Accuracy** during training:

### **VGG19 Accuracy Plot**  
![VGG19 Accuracy](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/TRAIN-VGG19.png)

### **VGG19 Loss Plot**  
![VGG19 Loss](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/Loss_VGG19.png)


---

### **Observations**:
- **VGG19 with CHALE** achieved the highest performance with a **train accuracy of 97.46%**, **test accuracy of 92.56%**, and **validation accuracy of 92.24%**.
- **VGG19** showed strong results but did not match the performance of VGG19 with CHALE, with a **train accuracy of 95.54%** and **test accuracy of 91.65%**.
- **The model trained from scratch** performed well but did not reach the accuracy levels of VGG19 or VGG19 with CHALE, achieving a **train accuracy of 91.91%** and **test accuracy of 85.87%**.
- **VGG16** exhibited significantly lower performance, particularly in test accuracy (**79.89%**) and validation accuracy (**77.61%**), reinforcing the need for deeper architectures in medical imaging tasks.


---

### **Why VGG19 with CHALE Outperformed**:
- **Enhanced Feature Extraction**: The integration of the CHALE technique allowed VGG19 to enhance the visibility of diagnostic features in chest X-ray images, significantly improving the model's ability to learn relevant patterns.
- **Deep Learning Synergy**: The combination of VGG19’s deeper architecture and CHALE's image processing capabilities resulted in a powerful synergy, leading to better overall model performance on complex medical imaging tasks.

Despite resource constraints, VGG19 with CHALE delivered the highest accuracy and reliability, solidifying its position as the optimal choice for our classification task. With further fine-tuning and improved resources, we anticipate even greater advancements in performance.


---

# Model Evaluation Metrics 📊

In this section, we present the evaluation metrics for our model using the VGG19 architecture. These metrics help us understand the model's performance on the dataset, particularly in terms of classification accuracy.

---

## Classification Report 📈

The following table summarizes the precision, recall, F1-score, and support for each class in our dataset:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| 0.0   | 0.98      | 0.97   | 0.98     | 491    |
| 1.0   | 0.96      | 0.89   | 0.93     | 602   |
| 2.0   | 0.90      | 0.96   | 0.93     | 650   |
| 3.0   | 0.86      | 0.87   | 0.87     | 301    |
| 4.0   | 0.83      | 0.86   | 0.84     | 301    |
| 5.0   | 1.00      | 0.99   | 0.99     | 250    |
| **Accuracy** |  |  | **0.93** | **2595** |
| **Macro Avg** | 0.92 | 0.92 | 0.92 | 2595 |
| **Weighted Avg** | 0.93 | 0.93 | 0.93 | 2595 |

### Definitions:
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all actual positives.
- **F1-Score**: The weighted average of Precision and Recall. It ranges from 0 to 1, where 1 indicates perfect precision and recall.
- **Support**: The number of actual occurrences of the class in the specified dataset.

---

## Confusion Matrix 📉

The confusion matrix summarizes the prediction results of a classification problem. Below is the confusion matrix for the VGG19 model:

![Confusion Matrix](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/Confusion%20Matrix_VGG19.png)

### Notes:
- The confusion matrix allows us to visualize how many instances were classified correctly and incorrectly across different classes.
- Diagonal values represent the correct predictions for each class, while off-diagonal values represent misclassifications.

---
<h2 id="#grad-cam-visualizations"> Grad-CAM Visualization for X-Ray Classes 📊</h2>

In this section, we present examples of the X-ray images for each class that will be analyzed using Grad-CAM. The images illustrate how the model identifies specific patterns related to different respiratory diseases.

| Class                | Image                                                                                     |
|----------------------|-------------------------------------------------------------------------------------------|
| **Normal**           | ![Normal Class](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/Normal_GradCAM.png)  |
| **Pneumonia-Bacterial** | ![Pneumonia Class](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/Pneumonia-Bacterial_GradCAM.png) |
| **COVID-19**        | ![COVID-19 Class](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/COVID-19_GradCAM.png)      |
| **Pneumonia-Viral**        | ![COVID-19 Class](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/Pneumonia-Viral_GradCAM.png)      |
| **Tuberculosis**    | ![Tuberculosis Class](https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/Tuberculosis_GradCAM.png)  |

---

# Demo Video 🎥

Watch this demo video to see how to use the website effectively:

<a href="https://drive.google.com/file/d/165S2Kf6V3nmRi5BGgcnp9tlOHN8B9xWP/view?usp=sharing">
    <img src="https://img.youtube.com/vi/VIDEO_ID/0.jpg" alt="Demo Video" width="640" height="360">
</a>

---

## Deployment Link 🌐
Explore the deployed version of our Chest Disease Detection by X-ray web application [here](https://github.com/fatma2123456/BreatheAI-Website/blob/main/README.md).

---
## Authors ✨
<pre>
Fatma Elzhra ahmed  - Artificial Intelligence Engineering - <b><a href="https://github.com/fatma2123456">fatma2123456</a></b>
Abdelrahman Mohamed - Computer Science Engineering - <b><a href="https://github.com/AbdelrahmanMohamed252">AbdelrahmanMohamed252</a></b>
Hanin Mustafa  - Computer Systems Engineering  - <b><a href="https://github.com/HaninMustafa9">HaninMustafa9</a></b>
Karim Ehab   - Communication and computer engineering - <b><a href="https://github.com/Eng-Karim-Ehab">Eng-Karim-Ehab</a></b>
Mahmoud Anas - Computer Science - <b><a href="https://github.com/MahmoudAnas046">MahmoudAnas046</a></b>
 
Supervised By :
Eng / Mahmoud Talaat 
Ai Engineer at MCiT (Ministry of Communication and Information Technology)
TA at Zewail University ( Artificial intelligence and Data Science )
  <div style="text-align: right;">
                                                                                                     <img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/Image/Picture2_20241025_195821_0000.png" alt="Government Logo" width="150"/> 
</div>
</pre>
